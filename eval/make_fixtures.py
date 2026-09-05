"""Generate binary corpus fixtures that cannot be stored as plain text.

Currently produces a deterministic two-page PDF whose answer sentences straddle
the page boundary, which is the fixture Phase 5 (structural chunking) must fix.

Run:  python -m eval.make_fixtures
"""

from __future__ import annotations

from pathlib import Path


CORPUS = Path(__file__).resolve().parent / "corpus"

PAGE_ONE = [
    "Kestrel Logistics - Annual Operations Review 2026",
    "",
    "Section 1. Network summary",
    "",
    "The European network handled 4.18 million shipments during the reporting",
    "year, an increase of eleven percent over the previous year. Hamburg",
    "remained the largest hub with 1.32 million shipments, followed by Rotterdam",
    "with 0.91 million and Lyon with 0.64 million.",
    "",
    "Section 2. Forecast accuracy",
    "",
    "The demand-forecasting platform supplied by Northwind Analytics reached a",
    "weighted mean absolute percentage error of 8.4 percent across all",
    "warehouses, against a target of 10 percent. Accuracy was weakest in the",
    "Lyon region during the second quarter.",
    "",
    "Section 3. Cold chain incident",
    "",
    "On 14 August 2026 the refrigeration unit at the Rotterdam cross-dock failed",
    "during a heat warning. The root cause of the cold chain failure was a",
    "blocked condenser coil that had been flagged in the June inspection but was",
    "not scheduled for cleaning, because the maintenance backlog was ranked by",
]

PAGE_TWO = [
    "asset age rather than by inspection severity. Total spoilage was recorded",
    "at EUR 312,000 and the insurer settled 78 percent of the claim.",
    "",
    "Section 4. Corrective actions",
    "",
    "The maintenance backlog is now ranked by inspection severity first and",
    "asset age second. A second condenser was installed at Rotterdam in October",
    "2026 to remove the single point of failure.",
    "",
    "Section 5. Headcount",
    "",
    "Planning headcount grew from 41 to 47 full-time equivalents. Two roles",
    "remain open in the Lyon planning team as of 31 December 2026.",
    "",
    "Section 6. Outlook",
    "",
    "Volume growth of six to eight percent is expected for 2027, concentrated in",
    "the Iberian corridor. No additional hub investment is planned before the",
    "second half of the year.",
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 11 Tf", "72 720 Td", "15 TL"]
    for line in lines:
        parts.append(f"({_escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []

    page_object_ids = [3 + 2 * i for i in range(len(pages))]
    font_id = 3 + 2 * len(pages)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_object_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1"))

    for index, lines in enumerate(pages):
        page_id = page_object_ids[index]
        content_id = page_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode("latin-1")
        )
        stream = _content_stream(lines)
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("latin-1")
    return bytes(out)


FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# One line per sentence, in logical order - the order a reader reads them in and
# the order extraction has to return them in.
PERSIAN_LINES = [
    "قرارداد خدمات فنی شماره ۱۳۷",
    "مبلغ کل قرارداد ۲٬۴۰۰٬۰۰۰٬۰۰۰ ریال است.",
    "قرارداد در تاریخ ۱۴۰۶/۰۱/۱۴ خاتمه می‌یابد.",
]
# DejaVu is the font on this image that covers both Persian letters and Persian
# digits; the builder falls back to any font that does.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]


def persian_font_path() -> str | None:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def build_persian_pdf(right_to_left: bool) -> bytes | None:
    """A Persian PDF laid out the way a real producer would lay it out.

    `right_to_left` picks which kind of producer: one that reorders the glyphs
    for display, or one that simply places them in logical order. Extraction
    fails differently on each, which is the whole point of having both.
    Returns None when the machine has no font with Persian glyphs.
    """
    try:
        import pymupdf
    except ImportError:
        return None
    font_path = persian_font_path()
    if font_path is None:
        return None

    document = pymupdf.open()
    page = document.new_page()
    font = pymupdf.Font(fontfile=font_path)
    writer = pymupdf.TextWriter(page.rect)
    y = 100.0
    for line in PERSIAN_LINES:
        width = font.text_length(line, 14)
        x = page.rect.width - 72 - width if right_to_left else 72
        writer.append((x, y), line, font=font, fontsize=14, right_to_left=right_to_left)
        y += 30
    writer.write_text(page)
    return document.tobytes()


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    target = CORPUS / "annual-review.en.pdf"
    target.write_bytes(build_pdf([PAGE_ONE, PAGE_TWO]))
    print(f"wrote {target} ({target.stat().st_size} bytes)")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for rtl, name in ((True, "persian-visual-order.pdf"), (False, "persian-logical-order.pdf")):
        data = build_persian_pdf(rtl)
        if data is None:
            print(f"skipped {name}: PyMuPDF or a Persian-capable font is missing")
            continue
        path = FIXTURES / name
        path.write_bytes(data)
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
