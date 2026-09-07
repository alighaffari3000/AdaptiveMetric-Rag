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
        blocks: list[tuple[str, int | None, str | None]] = []
        stack: list[tuple[int, str]] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            level = _docx_heading_level(paragraph)
            if level is not None:
                stack = [(lv, t) for lv, t in stack if lv < level] + [(level, text)]
            # A heading is also the first line of its own section, under its own
            # path. Every other format keeps it in the text, and without it a
            # term that appears only in a heading is missing from the BM25
            # postings for Word documents alone. Packing joins it to the body
            # below; not stranding it is `chunk_blocks`' job, for every format.
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


Block = tuple[str, "int | None", "str | None"]


def _units(blocks: list[Block]) -> list[list[Block]]:
    """Group each run of text-less headings with the block it opens.

    A heading names what comes after it, so it travels with that text and not
    with the section before. Doing this before packing rather than during it is
    what keeps a heading from ever being left alone: packing then only ever
    sees units that already carry their own body.

    A trailing run with nothing after it joins the unit before, since there is
    nothing else to attach it to. A document of nothing but headings is one
    unit, and `size` is all that bounds it.
    """
    units: list[list[Block]] = []
    opening: list[Block] = []
    for text, page, section in blocks:
        if structure.heading_only(text, section):
            opening.append((text, page, section))
            continue
        units.append(opening + [(text, page, section)])
        opening = []
    if opening:
        if units:
            units[-1].extend(opening)
        else:
            units.append(opening)
    return units


def _describe(parts: list[Block]) -> dict:
    """The chunk a group of blocks makes: its text, sections and page."""
    covered = list(dict.fromkeys(section for _, _, section in parts if section))
    return {
        "content": "\n".join(text for text, _, _ in parts).strip(),
        "page": next((page for _, page, _ in parts if page is not None), None),
        "section": (covered[0].split(structure.PATH_SEPARATOR)[-1] if covered else None),
        "section_path": structure.merge_section_paths(covered),
        "sections": covered,
    }


def _windows(parts: list[Block], size: int, overlap: int) -> list[dict]:
    """Cut a unit too long for one chunk, labelling each window by its span.

    Each block occupies a known range of the joined text, so a window declares
    the sections it actually overlaps and is cited by the page that supplies
    most of its characters. Reading the labels off the whole unit instead let a
    window of pure heading text claim the body section beneath it.
    """
    spans: list[tuple[int, int, Block]] = []
    cursor = 0
    for part in parts:
        spans.append((cursor, cursor + len(part[0]), part))
        cursor += len(part[0]) + 1
    text = "\n".join(part[0] for part in parts)

    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end),
                           text.rfind("؟", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            here = [(min(end, stop) - max(start, begin), block)
                    for begin, stop, block in spans if begin < end and stop > start]
            covered = list(dict.fromkeys(block[2] for _, block in here if block[2]))
            page = next((block[1] for _, block in sorted(here, key=lambda item: -item[0])
                         if block[1] is not None), None)
            chunks.append({
                "content": piece,
                "page": page,
                "section": (covered[0].split(structure.PATH_SEPARATOR)[-1] if covered else None),
                "section_path": structure.merge_section_paths(covered),
                "sections": covered,
            })
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_blocks(blocks: list[Block], size: int, overlap: int) -> list[dict]:
    """Cut blocks into chunks that follow the document's own divisions.

    `extract` emits one block per section. Sections shorter than `size` are
    packed together, because a chunk per heading shreds a short document:
    measured on the golden set, per-section chunks tripled the chunk count and
    cost 8 points of MRR, all of it granularity rather than structure
    (eval/BASELINE.md).

    `size` is the only bound on packing, and a chunk carries the list of every
    section it covers. Refusing to pack across a chapter boundary was tried and
    measured: it triples the chunk count on this corpus and costs 5.7 points of
    hit@5, taking the page-boundary fixture back to zero. It also buys less
    than it appears to, because packing only ever merges sections smaller than
    `size` - in a document whose chapters are long enough for the distinction
    to matter, each chapter is split on its own anyway.

    Packing works on units, not blocks: a heading is glued to the text it opens
    before any of this runs, so no arrangement of sizes can leave one stranded
    in a chunk of its own or on the end of the section before it.
    """
    chunks: list[dict] = []
    pending: list[Block] = []

    def flush() -> None:
        if pending:
            described = _describe(pending)
            if described["content"]:
                chunks.append(described)
            pending.clear()

    def length(parts: list[Block]) -> int:
        return sum(len(text) + 1 for text, _, _ in parts)

    for unit in _units(blocks):
        unit = [(re.sub(r"[ \t]+", " ", text).strip(), page, section) for text, page, section in unit]
        unit = [part for part in unit if part[0]]
        if not unit:
            continue
        if length(unit) > size:
            flush()
            chunks.extend(_windows(unit, size, overlap))
            continue
        if length(pending) + length(unit) > size:
            flush()
        pending.extend(unit)
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
