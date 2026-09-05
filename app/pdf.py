"""Reading a PDF, including a Persian one, without scrambling it.

`pypdf` returns a page as one undifferentiated string. That is enough to index
words and not enough to chunk well: headings, tables and the reading order of a
two-column page all disappear, and a chunk boundary lands wherever the character
count runs out. PyMuPDF exposes position, font size and table structure, so this
module reads with PyMuPDF and keeps `pypdf` for two jobs it is still better at.

The Persian problem is worth stating plainly, because the obvious fix is wrong.
Text in a PDF is positioned glyphs with no direction of its own. Whether the
producer already reordered them for display, and whether the extractor reorders
them again, are independent, and the four combinations were measured on
fixtures built both ways (`eval/BASELINE.md`):

- producer laid out right-to-left: neither extractor mirrors the words, but
  PyMuPDF reverses digit runs, so ۱۳۷ is read as ۷۳۱ - silently the wrong number.
- producer laid out left-to-right: both extractors mirror every word, PyMuPDF at
  the character level and pypdf at the word level too.

So neither engine is "the right one". This module detects mirrored text by
counting common Persian words against their reversals, repairs it by
un-mirroring, and repairs reversed digit runs by cross-checking them against the
other engine, which never reverses them.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .text import language_of, looks_mirrored, unmirror

logger = logging.getLogger("adaptive_metric_rag.pdf")

OCR_LANGUAGES = "fas+eng"
OCR_DPI = 300
OCR_TIMEOUT_SECONDS = 120
# A heading is set larger than the body. 1.15 keeps a slightly emphasised run of
# body text out, while catching the smallest step a document usually makes.
HEADING_RATIO = 1.15
HEADING_MAX_CHARS = 120


@dataclass
class Block:
    """One structural unit of a document, as extracted."""

    text: str
    page: int | None = None
    heading_level: int = 0  # 0 is body text; 1 and up are headings
    kind: str = "text"      # "text", "heading" or "table"


@dataclass
class Extraction:
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    engine: str = "pymupdf"


def _load_pymupdf():
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - exercised only without the wheel
        return None
    # PyMuPDF prints advice about an optional layout package straight to stdout
    # on every table scan, which would litter API logs and evaluation output.
    try:
        pymupdf.set_messages(pylogging_name=f"{logger.name}.mupdf", pylogging_level=logging.DEBUG)
    except Exception:  # pragma: no cover - older builds have no message routing
        pass
    return pymupdf


def _pypdf_pages(payload: bytes) -> list[str]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(payload))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # a cross-check must never fail the ingestion
        logger.warning("pypdf cross-check failed: %r", exc)
        return []


_DIGIT = "0-9۰-۹٠-٩"
# A number keeps the separators inside it: a bidi pass reverses ۲٬۴۰۰٬۰۰۰٬۰۰۰ and
# ۱۴۰۶/۰۱/۱۴ as single units, so the unit has to be recognised as one.
_NUMBER_RUN = re.compile(rf"[{_DIGIT}][{_DIGIT}٬٫,./-]*[{_DIGIT}]|[{_DIGIT}]")


def align_digit_runs(text: str, reference: str) -> str:
    """Restore numbers a bidi pass reversed, using the other engine as truth.

    A run is only rewritten when the reference contains exactly its reversal and
    not the run itself, so an amount that genuinely differs between the two
    readings is left alone, and a palindrome is left alone by construction.
    """
    reference_runs = set(_NUMBER_RUN.findall(reference))
    if not reference_runs:
        return text

    def fix(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(run) > 1 and run not in reference_runs and run[::-1] in reference_runs:
            return run[::-1]
        return run

    return _NUMBER_RUN.sub(fix, text)


def page_direction(page_text: str, reference: str) -> str:
    """How this page has to be read: as extracted, un-mirrored, or from pypdf.

    Decided once per page rather than per block. A block is a few words long,
    and the evidence for direction - how many common Persian words read forwards
    against backwards - is too thin at that size to be trusted.
    """
    if language_of(page_text) != "fa":
        return "as-is"
    if looks_mirrored(page_text):
        # The other engine read the same page the right way round: take its
        # text, losing this page's structure to keep its words.
        return "reference" if reference and not looks_mirrored(reference) else "unmirror"
    return "as-is"


def repair_direction(text: str, reference: str = "", policy: str = "") -> str:
    """Apply the page's reading decision to one block of it."""
    if not policy:
        policy = page_direction(text, reference)
    if policy == "unmirror":
        text = unmirror(text)
    elif policy == "reference":
        return text  # the caller replaces the whole page; a block cannot
    if reference and not looks_mirrored(reference):
        text = align_digit_runs(text, reference)
    return text


def _ocr_page(pymupdf_page, warnings: list[str]) -> str:
    """Read a scanned page, if the machine has an OCR engine installed."""
    if shutil.which("tesseract") is None:
        message = ("A page carries no text layer and tesseract is not installed, so it was "
                   "skipped. Install tesseract with the 'fas' language data to index scanned pages.")
        if message not in warnings:
            warnings.append(message)
        return ""
    image = pymupdf_page.get_pixmap(dpi=OCR_DPI).tobytes("png")
    try:
        completed = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", OCR_LANGUAGES],
            input=image, capture_output=True, timeout=OCR_TIMEOUT_SECONDS, check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        warning = f"OCR failed on a page without a text layer: {type(exc).__name__}"
        if warning not in warnings:
            warnings.append(warning)
        return ""
    return completed.stdout.decode("utf-8", errors="replace")


def _serialize_table(table) -> str:
    """A table as one line per row, so a row survives chunking as a unit."""
    rows = table.extract()
    if not rows:
        return ""
    header = [str(cell).strip() if cell else "" for cell in rows[0]]
    lines = []
    for row in rows[1:]:
        cells = [str(cell).strip() if cell else "" for cell in row]
        if not any(cells):
            continue
        pairs = [f"{name}: {value}" if name else value
                 for name, value in zip(header + [""] * len(cells), cells) if value]
        lines.append(" | ".join(pairs))
    if not lines:  # a header-only table is still worth indexing
        lines = [" | ".join(cell for cell in header if cell)]
    return "\n".join(lines)


def _line_text(line: dict[str, Any]) -> tuple[str, float, bool]:
    text = "".join(span.get("text", "") for span in line.get("spans", []))
    sizes = [span.get("size", 0.0) for span in line.get("spans", []) if span.get("text", "").strip()]
    bold = all(bool(span.get("flags", 0) & 16) for span in line.get("spans", []) if span.get("text", "").strip())
    return text, max(sizes, default=0.0), bold


def _heading_levels(sizes: list[float], body_size: float) -> dict[float, int]:
    """Rank the sizes above body text: the largest is level 1, the next level 2."""
    distinct = sorted({round(size, 1) for size in sizes if size > body_size * HEADING_RATIO}, reverse=True)
    return {size: level for level, size in enumerate(distinct[:4], start=1)}


def extract_pdf(payload: bytes) -> Extraction:
    """Blocks in reading order, with page numbers, headings and tables."""
    pymupdf = _load_pymupdf()
    if pymupdf is None:
        pages = _pypdf_pages(payload)
        return Extraction(
            blocks=[Block(text=text, page=number) for number, text in enumerate(pages, 1) if text.strip()],
            warnings=["PyMuPDF is not installed; the PDF was read without structure."],
            engine="pypdf",
        )

    warnings: list[str] = []
    reference_pages = _pypdf_pages(payload)
    document = pymupdf.open(stream=payload, filetype="pdf")
    try:
        raw_pages = [page.get_text("dict") for page in document]
        all_sizes = [size for page in raw_pages for block in page.get("blocks", [])
                     for line in block.get("lines", []) for size in [_line_text(line)[1]] if size]
        body_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0.0
        levels = _heading_levels(all_sizes, body_size)

        blocks: list[Block] = []
        for number, page in enumerate(document, start=1):
            reference = reference_pages[number - 1] if number <= len(reference_pages) else ""
            policy = page_direction(page.get_text(), reference)
            if policy == "reference":
                warnings.append(f"Page {number} was read in visual order; it was re-read without "
                                "structure so the Persian words stay whole.")
                blocks.append(Block(text=align_digit_runs(reference, reference), page=number))
                continue
            table_rectangles = []
            for table in _find_tables(page):
                serialized = _serialize_table(table)
                if serialized:
                    table_rectangles.append(table.bbox)
                    blocks.append(Block(text=repair_direction(serialized, reference, policy),
                                        page=number, kind="table"))
            page_blocks = _page_blocks(raw_pages[number - 1], table_rectangles, levels, body_size)
            if not page_blocks and not table_rectangles:
                scanned = _ocr_page(page, warnings)
                if scanned.strip():
                    blocks.append(Block(text=repair_direction(scanned), page=number))
                continue
            for text, level in page_blocks:
                blocks.append(Block(text=repair_direction(text, reference, policy), page=number,
                                    heading_level=level, kind="heading" if level else "text"))
        return Extraction(blocks=blocks, warnings=warnings)
    finally:
        document.close()


def _find_tables(page) -> list:
    try:
        # Table detection prints advice about an optional package on every call,
        # straight to stdout and not through PyMuPDF's message routing, which
        # would litter both the API logs and the evaluation output.
        with contextlib.redirect_stdout(io.StringIO()):
            return list(page.find_tables().tables)
    except Exception as exc:  # table detection is a bonus, never a blocker
        logger.debug("table detection skipped on a page: %r", exc)
        return []


def _overlaps(rectangle: tuple[float, float, float, float],
              others: list[tuple[float, float, float, float]]) -> bool:
    x0, y0, x1, y1 = rectangle
    for ox0, oy0, ox1, oy1 in others:
        if x0 < ox1 and ox0 < x1 and y0 < oy1 and oy0 < y1:
            return True
    return False


def _page_blocks(raw_page: dict[str, Any], table_rectangles: list, levels: dict[float, int],
                 body_size: float) -> list[tuple[str, int]]:
    """Group a page's lines into blocks, splitting wherever a heading appears."""
    collected: list[tuple[str, int]] = []
    for block in raw_page.get("blocks", []):
        if block.get("type") != 0 or _overlaps(block.get("bbox", (0, 0, 0, 0)), table_rectangles):
            continue
        paragraph: list[str] = []
        for line in block.get("lines", []):
            text, size, bold = _line_text(line)
            if not text.strip():
                continue
            level = levels.get(round(size, 1), 0)
            if not level and bold and body_size and len(text.strip()) <= HEADING_MAX_CHARS:
                level = len(levels) + 1
            if level:
                if paragraph:
                    collected.append(("\n".join(paragraph), 0))
                    paragraph = []
                collected.append((text.strip(), level))
            else:
                paragraph.append(text.strip())
        if paragraph:
            collected.append(("\n".join(paragraph), 0))
    return collected
