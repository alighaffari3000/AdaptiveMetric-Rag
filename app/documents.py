from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from . import database
from .chunking import Piece, Window, build_pieces
from .embeddings import create_embeddings, document_embedding_text
from .index import index
from .models import AppSettings
from .pdf import Block, Extraction, extract_pdf
from .text import tokenize


ALLOWED = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html", ".htm"}


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _markdown_blocks(text: str) -> list[Block]:
    """`#` levels become the section path; everything else is a paragraph."""
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        joined = "\n".join(paragraph).strip()
        if joined:
            blocks.append(Block(text=joined))
        paragraph.clear()

    for line in text.splitlines():
        heading = _MARKDOWN_HEADING.match(line)
        if heading:
            flush()
            blocks.append(Block(text=heading.group(2), heading_level=len(heading.group(1)), kind="heading"))
        elif line.strip():
            paragraph.append(line.strip())
        else:
            flush()
    flush()
    return blocks


def _html_blocks(text: str) -> list[Block]:
    soup = BeautifulSoup(text, "html.parser")
    for unwanted in soup(["script", "style"]):
        unwanted.decompose()
    blocks: list[Block] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "pre", "blockquote"]):
        content = element.get_text(" ", strip=True)
        if not content:
            continue
        if element.name.startswith("h") and element.name[1:].isdigit():
            blocks.append(Block(text=content, heading_level=int(element.name[1:]), kind="heading"))
        else:
            blocks.append(Block(text=content))
    if not blocks:  # a document with no structural tags at all
        flat = soup.get_text("\n", strip=True)
        blocks = [Block(text=flat)] if flat else []
    return blocks


def _docx_blocks(payload: bytes) -> list[Block]:
    document = DocxDocument(io.BytesIO(payload))
    blocks: list[Block] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = paragraph.style.name if paragraph.style else ""
        if style.startswith("Heading"):
            level = int(style.split()[-1]) if style.split()[-1].isdigit() else 1
            blocks.append(Block(text=text, heading_level=level, kind="heading"))
        else:
            blocks.append(Block(text=text))
    for table in document.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        header, *body = rows
        lines = [" | ".join(f"{name}: {value}" if name else value
                            for name, value in zip(header, row) if value)
                 for row in body if any(row)]
        if lines:
            blocks.append(Block(text="\n".join(lines), kind="table"))
    return blocks


def _csv_blocks(text: str) -> list[Block]:
    """One line per row, each cell labelled, so a row survives chunking whole."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header, *body = rows
    if not body:
        return [Block(text=" | ".join(header), kind="table")]
    lines = [" | ".join(f"{name}: {value}" if name else value
                        for name, value in zip(header + [""] * len(row), row) if value)
             for row in body if any(cell.strip() for cell in row)]
    return [Block(text="\n".join(lines), kind="table")] if lines else []


def extract(filename: str, payload: bytes) -> Extraction:
    """Read a file into blocks that remember their structure."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        return extract_pdf(payload)
    if suffix == ".docx":
        return Extraction(blocks=_docx_blocks(payload), engine="python-docx")

    text = payload.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return Extraction(blocks=_html_blocks(text), engine="beautifulsoup")
    if suffix == ".md":
        return Extraction(blocks=_markdown_blocks(text), engine="markdown")
    if suffix == ".json":
        return Extraction(blocks=[Block(text=json.dumps(json.loads(text), ensure_ascii=False, indent=2))],
                          engine="json")
    if suffix == ".csv":
        return Extraction(blocks=_csv_blocks(text), engine="csv")
    return Extraction(blocks=[Block(text=text)], engine="text")


def chunk_document(filename: str, payload: bytes, settings: AppSettings) -> tuple[list[Piece], list[Window], list[str]]:
    """Extraction and chunking together: the CPU-bound half of ingestion."""
    extraction = extract(filename, payload)
    pieces, windows = build_pieces(extraction.blocks, settings.child_tokens,
                                   settings.child_overlap_tokens, settings.parent_tokens)
    return pieces, windows, extraction.warnings


def content_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def find_duplicate(digest: str) -> dict | None:
    """An identical file uploaded twice used to produce two full sets of chunks."""
    for document in database.rows("SELECT id,name,created_at,metadata FROM documents"):
        try:
            metadata = json.loads(document["metadata"] or "{}")
        except json.JSONDecodeError:
            continue
        if metadata.get("sha256") == digest:
            return document
    return None


def refresh_chunk_tokens() -> int:
    """Recompute every chunk's stored tokens with the current normalizer.

    Returns the number of chunks whose tokens actually changed, so the caller
    can tell an upgrade from a no-op.
    """
    stored = database.rows("SELECT id,content,tokens FROM chunks")
    updates = []
    for chunk in stored:
        tokens = database.json_value(tokenize(chunk["content"]))
        if tokens != chunk["tokens"]:
            updates.append((tokens, chunk["id"]))
    if updates:
        with database.connect() as db:
            db.executemany("UPDATE chunks SET tokens=? WHERE id=?", updates)
        index.invalidate()
    return len(updates)


def parent_windows(chunks: list[dict]) -> list[dict]:
    """Widen retrieved children to the windows an answer can be written from.

    Retrieval scores small chunks, generation reads large ones. Children of the
    same parent collapse into one window, in the order their best child ranked,
    so the model is not handed the same paragraph three times.
    """
    ordered: list[str] = []
    for chunk in chunks:
        parent_id = chunk.get("parent_id")
        if parent_id and parent_id not in ordered:
            ordered.append(parent_id)
    stored = {}
    if ordered:
        placeholders = ",".join("?" * len(ordered))
        stored = {row["id"]: row for row in
                  database.rows(f"SELECT id,content,page_start,page_end,section_path FROM parents "
                                f"WHERE id IN ({placeholders})", tuple(ordered))}
    widened: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        parent = stored.get(chunk.get("parent_id") or "")
        if parent is None:
            widened.append(chunk)
            continue
        if parent["id"] in seen:
            continue
        seen.add(parent["id"])
        window = dict(chunk)
        window["content"] = parent["content"]
        window["page"] = parent["page_start"] if parent["page_start"] is not None else chunk.get("page")
        widened.append(window)
    return widened


async def ingest(filename: str, content_type: str, payload: bytes, settings: AppSettings,
                 document_id: str = "", progress=None) -> dict:
    """Parse, chunk, embed and store one document.

    `progress` is called with a fraction between 0 and 1 so a long PDF can
    report where it is instead of holding an HTTP request open in silence.
    """
    doc_id = document_id or uuid.uuid4().hex

    def report(fraction: float) -> None:
        if progress is not None:
            progress(fraction)

    # Parsing and chunking are CPU bound; keep them out of the event loop.
    pieces, windows, warnings = await asyncio.to_thread(chunk_document, filename, payload, settings)
    report(.35)
    vectors = await create_embeddings(
        settings,
        [document_embedding_text(filename, piece.section_path or piece.section, piece.content)
         for piece in pieces],
    )
    report(.85)
    now = datetime.now(timezone.utc).isoformat()
    metadata = database.json_value({"sha256": content_digest(payload), "warnings": warnings})
    parent_ids = [uuid.uuid4().hex for _ in windows]
    with database.connect() as db:
        db.execute(
            "INSERT INTO documents(id,name,type,size,chunks,created_at,metadata,status,progress,error) "
            "VALUES(?,?,?,?,?,?,?,'ready',1.0,'') "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name,type=excluded.type,size=excluded.size,"
            "chunks=excluded.chunks,metadata=excluded.metadata,status='ready',progress=1.0,error=''",
            (doc_id, filename, content_type or "application/octet-stream", len(payload), len(pieces), now, metadata),
        )
        for position, (window, parent_id) in enumerate(zip(windows, parent_ids)):
            db.execute(
                "INSERT INTO parents(id,document_id,position,content,page_start,page_end,section_path) "
                "VALUES(?,?,?,?,?,?,?)",
                (parent_id, doc_id, position, window.content, window.page_start, window.page_end,
                 window.section_path),
            )
        for position, (piece, vector) in enumerate(zip(pieces, vectors)):
            db.execute(
                "INSERT INTO chunks(id,document_id,position,page,page_end,section,section_path,parent_id,"
                "content,vector,tokens,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, doc_id, position, piece.page_start, piece.page_end, piece.section,
                 piece.section_path, parent_ids[piece.parent] if parent_ids else None, piece.content,
                 database.encode_vector(vector), database.json_value(tokenize(piece.content)), "{}"),
            )
    index.invalidate()
    report(1.0)
    return {"id": doc_id, "name": filename, "type": content_type, "size": len(payload),
            "chunks": len(pieces), "parents": len(windows), "created_at": now,
            "warnings": warnings, "status": "ready"}
