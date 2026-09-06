"""Structure extraction: headings, section paths, and section-aware chunking.

The property under test throughout is the one STAIR (arXiv:2609.03874) argues
for: a chunk is a whole number of the document's own sections, and it says
which ones. A chunk that silently spans half of one heading and half of the
next is the failure these tests exist to catch.
"""

import io

import pytest

from app.documents import chunk_blocks, extract
from app.structure import (
    PATH_SEPARATOR,
    build_toc,
    detect,
    detect_html,
    detect_markdown,
    detect_plain,
    merge_section_paths,
    section_paths,
)


def paths_of(chunks):
    return [chunk["section_path"] for chunk in chunks]


def test_markdown_headings_build_a_nested_path():
    text = "# Handbook\n\n## Leave\n\nintro\n\n### Sick leave\n\nup to eight days\n"
    headings = detect_markdown(text)
    assert [(h.level, h.title) for h in headings] == [
        (1, "Handbook"), (2, "Leave"), (3, "Sick leave")
    ]
    spans = section_paths(text, headings)
    deepest = spans[-1][2]
    assert deepest == "Handbook > Leave > Sick leave"


def test_a_deeper_heading_does_not_close_its_parent():
    text = "# A\n\n## B\n\n### C\n\ntext\n\n## D\n\nmore\n"
    spans = section_paths(text, detect_markdown(text))
    assert "A > B > C" in [path for _, _, path in spans]
    # D is a sibling of B, so C must not survive into it.
    assert "A > D" in [path for _, _, path in spans]


def test_text_before_the_first_heading_keeps_the_base_path():
    text = "preamble\n\n# Title\n\nbody\n"
    spans = section_paths(text, detect_markdown(text))
    assert spans[0][2] == ""
    assert text[spans[0][0]:spans[0][1]].strip() == "preamble"


def test_fenced_code_is_not_mistaken_for_headings():
    text = "# Real\n\n```\n# not a heading\n## also not\n```\n\nbody\n"
    assert [h.title for h in detect_markdown(text)] == ["Real"]


def test_persian_legal_numbering_is_detected_with_its_nesting():
    text = "فصل اول — کلیات\nمتن\nماده ۳ — معامله کوچک\nمتن\nتبصره ۱\nسه فروشنده\n"
    headings = detect_plain(text)
    assert [(h.level, h.title) for h in headings] == [
        (1, "فصل اول — کلیات"), (2, "ماده ۳ — معامله کوچک"), (3, "تبصره ۱")
    ]
    assert section_paths(text, headings)[-1][2] == "فصل اول — کلیات > ماده ۳ — معامله کوچک > تبصره ۱"


@pytest.mark.parametrize("line", [
    "ماده 12 — تعاریف",
    "Chapter 4",
    "Section 2. Forecast accuracy",
    "4.2.1 Rollback procedure",
])
def test_numbering_conventions_across_scripts(line):
    assert detect_plain(line + "\nbody\n"), f"{line!r} should read as a heading"


@pytest.mark.parametrize("line", [
    "مبلغ کل قرارداد ۲۴۰ میلیون ریال است و در چهار قسط پرداخت می‌شود",
    "In 2026 the network handled 4.18 million shipments across the region",
    "1 shipment was delayed,",
])
def test_prose_is_not_mistaken_for_a_heading(line):
    assert not detect_plain(line + "\n"), f"{line!r} is prose, not a heading"


def test_html_headings_survive_instead_of_being_flattened():
    html = "<h1>Release notes</h1><p>intro</p><h2>7.4.0</h2><p>scheduler change</p>"
    blocks = detect_html(html)
    assert ("intro", None, "Release notes") in blocks
    assert any(path == "Release notes > 7.4.0" for _, _, path in blocks)


def test_html_without_headings_still_yields_its_text():
    blocks = detect_html("<p>just a paragraph</p>")
    assert blocks and "just a paragraph" in blocks[0][0]
    assert blocks[0][2] == ""


def test_a_chunk_never_spans_two_sections_it_does_not_declare():
    blocks = [("first section body", None, "A > one"), ("second section body", None, "A > two")]
    packed = chunk_blocks(blocks, 900, 140)
    for chunk in packed:
        for word, section in (("first", "one"), ("second", "two")):
            if word in chunk["content"]:
                assert section in chunk["section_path"], \
                    f"chunk carries {word!r} text without declaring section {section!r}"


def test_packing_stops_at_the_chunk_size():
    blocks = [(f"section {index} " + "x" * 200, None, f"A > s{index}") for index in range(6)]
    packed = chunk_blocks(blocks, 500, 50)
    assert len(packed) > 1
    assert all(len(chunk["content"]) <= 500 for chunk in packed)


def test_a_section_longer_than_the_chunk_size_splits_and_keeps_its_path():
    blocks = [("word " * 500, None, "A > long")]
    packed = chunk_blocks(blocks, 400, 40)
    assert len(packed) > 1
    assert all(chunk["section_path"] == "A > long" for chunk in packed)


def test_a_packed_chunk_declares_every_section_it_covers():
    blocks = [("alpha", None, "Root > A"), ("beta", None, "Root > B"), ("gamma", None, "Root > C")]
    chunk = chunk_blocks(blocks, 900, 100)[0]
    assert chunk["section_path"] == "Root > A | B | C"
    assert build_toc([chunk["section_path"]]) == [
        {"path": "Root > A", "title": "A", "depth": 2},
        {"path": "Root > B", "title": "B", "depth": 2},
        {"path": "Root > C", "title": "C", "depth": 2},
    ]


def test_merging_shares_ancestry_once_and_keeps_a_single_path_intact():
    assert merge_section_paths(["A > B > x", "A > B > y"]) == "A > B > x | y"
    assert merge_section_paths(["A > B", "A > B"]) == "A > B"
    assert merge_section_paths(["A > x", "C > y"]) == "A > x | C > y"
    assert merge_section_paths(["", ""]) == ""


def test_a_structureless_document_chunks_exactly_as_before():
    text = "sentence. " * 300
    blocks = [(text, 1, None)]
    packed = chunk_blocks(blocks, 300, 50)
    assert len(packed) > 2
    assert all(chunk["page"] == 1 for chunk in packed)
    assert all(chunk["section_path"] == "" for chunk in packed)


def test_csv_and_json_carry_no_section_and_do_not_raise():
    for name, payload in (("t.csv", b"a,b\n1,2\n"), ("t.json", b'{"k": 1}')):
        chunks = chunk_blocks(extract(name, payload), 900, 100)
        assert chunks
        assert all(chunk["section_path"] == "" for chunk in chunks)


def test_markdown_ingest_produces_a_path_for_every_chunk():
    payload = ("# Guide\n\n## Leave\n\n### Sick\n\nup to eight days\n\n"
               "### Unpaid\n\nup to three months\n").encode("utf-8")
    chunks = chunk_blocks(extract("g.md", payload), 900, 100)
    assert chunks
    assert all(chunk["section_path"] for chunk in chunks)
    assert any("Sick" in chunk["section_path"] for chunk in chunks)


def test_pdf_uses_its_own_outline_in_preference_to_patterns():
    from eval.make_fixtures import build_pdf
    from pypdf import PdfReader, PdfWriter

    from app.structure import pdf_outline

    source = PdfReader(io.BytesIO(build_pdf([["Body of page one"], ["Body of page two"]])))
    writer = PdfWriter()
    for page in source.pages:
        writer.add_page(page)
    parent = writer.add_outline_item("Opening", 0)
    writer.add_outline_item("Nested detail", 1, parent=parent)
    buffer = io.BytesIO()
    writer.write(buffer)

    per_page = pdf_outline(PdfReader(io.BytesIO(buffer.getvalue())))
    assert per_page[1] == ["Opening"]
    assert per_page[2] == ["Opening", "Nested detail"]

    chunks = chunk_blocks(extract("outlined.pdf", buffer.getvalue()), 900, 100)
    assert any("Nested detail" in chunk["section_path"] for chunk in chunks)


def test_pdf_without_an_outline_falls_back_to_heading_patterns():
    from eval.make_fixtures import build_pdf

    payload = build_pdf([["Section 1. Network summary", "Shipments rose eleven percent."]])
    chunks = chunk_blocks(extract("plain.pdf", payload), 900, 100)
    assert any("Section 1" in chunk["section_path"] for chunk in chunks)


def test_a_malformed_outline_degrades_to_no_outline_rather_than_raising():
    from app.structure import pdf_outline

    class Broken:
        @property
        def outline(self):
            raise ValueError("corrupt outline")

    assert pdf_outline(Broken()) == {}


def test_detect_picks_the_right_strategy_per_suffix():
    markdown = "# Title\n\nbody\n"
    assert [h.title for h in detect(markdown, ".md")] == ["Title"]
    assert detect(markdown, ".csv") == []
    # A Markdown file numbering its sections in prose still gets a path.
    assert [h.title for h in detect("ماده ۱ — هدف\nمتن\n", ".md")] == ["ماده ۱ — هدف"]


def test_the_path_separator_is_not_produced_by_a_single_heading():
    text = "# Only\n\nbody\n"
    spans = section_paths(text, detect_markdown(text))
    assert PATH_SEPARATOR not in spans[-1][2]
