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
from typing import Any

from docx import Document as DocxDocument
from pypdf import PdfReader

from . import database, structure
from .embeddings import create_embeddings, document_embedding_text
from .index import index
from .models import AppSettings
from .retrieval import tokenize


ALLOWED = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".html", ".htm"}


def extract(filename: str, payload: bytes) -> list[tuple[str, int | None, str | None]]:
    """Split a file into (text, page, section_path) blocks.

    The third element used to be a single DOCX heading and was None for every
    other format. It is now the full section path for any format that carries
    one, which is what makes a chunk locatable inside its document.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(payload))
        outline = structure.pdf_outline(reader)
        blocks: list[tuple[str, int | None, str | None]] = []
        carried: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # A bookmark names the page it starts on; pages after it belong to
            # the same section until the next bookmark, so the path carries.
            carried = outline.get(number, carried)
            base = structure.PATH_SEPARATOR.join(carried)
            headings = structure.detect_plain(text) if not carried else []
            for start, end, path in structure.section_paths(text, headings, base):
                piece = text[start:end]
                if piece.strip():
                    blocks.append((piece, number, path or None))
        return blocks
    if suffix == ".docx":
        doc = DocxDocument(io.BytesIO(payload))
        blocks, stack = [], []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            level = _docx_heading_level(paragraph)
            if level is not None:
                stack = [(lv, t) for lv, t in stack if lv < level] + [(level, text)]
            # A heading is also the first line of its own section. Every other
            # format keeps it in the text; a heading-only term would otherwise
            # be missing from the BM25 postings for Word documents alone.
            blocks.append((text, None, structure.join_path(stack) or None))
        return blocks
    text = payload.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return [(body, page, path or None) for body, page, path in structure.detect_html(text)]
    if suffix == ".json":
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    elif suffix == ".csv":
        rows = csv.reader(io.StringIO(text))
        text = "\n".join(" | ".join(row) for row in rows)
    headings = structure.detect(text, suffix)
    return [(text[start:end], None, path or None) for start, end, path in
            structure.section_paths(text, headings) if text[start:end].strip()]


def _docx_heading_level(paragraph: Any) -> int | None:
    """The outline level of a Word heading style, or None for body text."""
    style = getattr(paragraph, "style", None)
    name = getattr(style, "name", "") or ""
    if name == "Title":
        return 1
    match = re.match(r"^Heading\s+(\d+)$", name)
    if not match:
        return None
    return min(int(match.group(1)), 6)


def chunk_blocks(blocks: list[tuple[str, int | None, str | None]], size: int, overlap: int) -> list[dict]:
    """Cut blocks into chunks that follow the document's own divisions.

    `extract` emits one block per section, so two things have to happen here.
    A section longer than `size` is split as before, and every piece keeps its
    path. Sections shorter than `size` are packed together, because a chunk per
    heading shreds a short document: measured on the golden set, per-section
    chunks tripled the chunk count and cost 8 points of MRR, all of it
    granularity rather than structure (eval/BASELINE.md).

    `size` is the only bound on packing, and a chunk carries the list of every
    section it covers. Refusing to pack across a chapter boundary was tried and
    measured: it triples the chunk count on this corpus and costs 5.7 points of
    hit@5, taking the page-boundary fixture back to zero (eval/BASELINE.md). It
    also buys less than it appears to, because packing only ever merges sections
    smaller than `size` - in a document whose chapters are long enough for the
    distinction to matter, each chapter is split on its own anyway.

    What keeps a packed chunk from being the blind window this was meant to
    replace is that it still begins and ends on a section boundary, and says
    which sections it holds.
    """
    chunks: list[dict] = []
    pending: list[tuple[str, int | None, str | None]] = []

    def flush() -> None:
        if not pending:
            return
        content = "\n".join(text for text, _, _ in pending).strip()
        if content:
            covered = list(dict.fromkeys(section for _, _, section in pending if section))
            page = next((page for _, page, _ in pending if page is not None), None)
            leaf = (pending[0][2] or "").split(structure.PATH_SEPARATOR)[-1] or None
            chunks.append({"content": content, "page": page, "section": leaf,
                           "section_path": structure.merge_section_paths(covered),
                           "sections": covered})
        pending.clear()

    def pending_length() -> int:
        return sum(len(text) + 1 for text, _, _ in pending)

    for text, page, section in blocks:
        clean = re.sub(r"[ \t]+", " ", text).strip()
        if not clean:
            continue
        if len(clean) > size:
            # A section too long to pack: emit what is pending, then window it.
            flush()
            start = 0
            while start < len(clean):
                end = min(len(clean), start + size)
                if end < len(clean):
                    boundary = max(clean.rfind("\n", start, end), clean.rfind(". ", start, end),
                                   clean.rfind("؟", start, end))
                    if boundary > start + size // 2:
                        end = boundary + 1
                piece = clean[start:end].strip()
                if piece:
                    leaf = section.split(structure.PATH_SEPARATOR)[-1] if section else None
                    chunks.append({"content": piece, "page": page, "section": leaf,
                                   "section_path": section or "",
                                   "sections": [section] if section else []})
                if end >= len(clean):
                    break
                start = max(start + 1, end - overlap)
            continue
        if pending_length() + len(clean) > size:
            flush()
        pending.append((clean, page, section))
    flush()
    return chunks


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


async def ingest(filename: str, content_type: str, payload: bytes, chunk_size: int, overlap: int,
                 settings: AppSettings) -> dict:
    doc_id = uuid.uuid4().hex
    # Parsing and chunking are CPU bound; keep them out of the event loop.
    pieces = await asyncio.to_thread(lambda: chunk_blocks(extract(filename, payload), chunk_size, overlap))
    vectors = await create_embeddings(
        settings,
        [document_embedding_text(filename, piece.get("section_path") or piece.get("section"), piece["content"])
         for piece in pieces],
    )
    now = datetime.now(timezone.utc).isoformat()
    toc = structure.build_toc([piece.get("sections") or [] for piece in pieces])
    metadata = database.json_value({"sha256": content_digest(payload), "toc": toc})
    with database.connect() as db:
        db.execute(
            "INSERT INTO documents(id,name,type,size,chunks,created_at,metadata) VALUES(?,?,?,?,?,?,?)",
            (doc_id, filename, content_type or "application/octet-stream", len(payload), len(pieces), now, metadata),
        )
        for position, (piece, vector) in enumerate(zip(pieces, vectors)):
            text = piece["content"]
            db.execute(
                "INSERT INTO chunks(id,document_id,position,page,section,section_path,content,vector,tokens,metadata) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, doc_id, position, piece["page"], piece["section"],
                 piece.get("section_path", ""), text,
                 database.encode_vector(vector), database.json_value(tokenize(text)),
                 # The display path is a label and cannot be parsed back into
                 # the sections it names; the list is the data.
                 database.json_value({"sections": piece.get("sections") or []})),
            )
    index.invalidate()
    return {"id": doc_id, "name": filename, "type": content_type, "size": len(payload), "chunks": len(pieces), "created_at": now}
