"""Phase 5: reading documents, including Persian PDFs, and chunking them."""

import pytest

from app.chunking import build_pieces, common_path, count_tokens, section_paths, sentences
from app.documents import extract
from app.pdf import Block, align_digit_runs, repair_direction
from app.text import looks_mirrored, unmirror

from eval.make_fixtures import PERSIAN_LINES, build_persian_pdf, build_pdf, PAGE_ONE, PAGE_TWO


# --- Persian PDFs: the same page, laid out the two ways producers lay it out ---


def persian_pdf(right_to_left: bool) -> bytes:
    data = build_persian_pdf(right_to_left)
    if data is None:
        pytest.skip("no PyMuPDF or no font with Persian glyphs on this machine")
    return data


@pytest.mark.parametrize("right_to_left", [True, False])
def test_a_persian_pdf_is_read_in_the_order_it_was_written(right_to_left):
    """Whichever way the producer laid the glyphs out, the words come back whole.

    Compared by codepoint, never by eye: a terminal reorders Persian for
    display, so a mirrored string can look perfectly correct while being wrong.
    """
    extraction = extract("contract.fa.pdf", persian_pdf(right_to_left))
    text = " ".join(block.text for block in extraction.blocks)
    for line in PERSIAN_LINES:
        for word in line.split():
            if any("؀" <= char <= "ۿ" for char in word) and not word[0].isdigit():
                assert word in text, f"{word!r} missing from {text!r}"


def test_a_persian_pdf_keeps_its_numbers_the_right_way_round():
    """۱۳۷ read as ۷۳۱ is not a formatting problem, it is the wrong number."""
    extraction = extract("contract.fa.pdf", persian_pdf(right_to_left=True))
    text = " ".join(block.text for block in extraction.blocks)
    assert "۱۳۷" in text
    assert "۱۴۰۶/۰۱/۱۴" in text


def mirror_like_an_extractor(text: str) -> str:
    """Reverse a line the way PyMuPDF does: words flipped, numbers left alone."""
    return unmirror(text)


def test_mirrored_text_is_detected_and_repaired():
    original = "قرارداد شماره ۱۳۷ در سال ۱۴۰۵ امضا شد"
    mirrored = mirror_like_an_extractor(original)
    assert looks_mirrored(mirrored)
    assert not looks_mirrored(original)
    # The repair is its own inverse for the shape an extractor produces.
    assert unmirror(mirrored) == original


def test_correct_text_is_left_alone():
    original = "مبلغ کل قرارداد ۲٬۴۰۰٬۰۰۰٬۰۰۰ ریال است"
    assert repair_direction(original, original) == original


def test_a_digit_run_is_only_flipped_when_the_other_engine_disagrees():
    assert align_digit_runs("شماره ۷۳۱ است", "شماره ۱۳۷ است") == "شماره ۱۳۷ است"
    # A genuinely different amount is not "corrected" into the other reading.
    assert align_digit_runs("شماره ۲۴۹ است", "شماره ۱۳۷ است") == "شماره ۲۴۹ است"


# --- structure ---


def test_markdown_headings_become_a_section_path():
    extraction = extract("policy.md", b"# Policy\n\nIntro line.\n\n## Leave\n\nTwenty six days.\n")
    levels = [(block.heading_level, block.text) for block in extraction.blocks]
    assert levels[0] == (1, "Policy")
    assert (2, "Leave") in levels
    paths = {block.text: path for block, path in section_paths(extraction.blocks)}
    assert paths["Twenty six days."] == ("Policy", "Leave")


def test_html_headings_become_a_section_path():
    html = b"<h1>Release</h1><p>Intro</p><h2>Fixes</h2><p>A bug was fixed.</p>"
    paths = {block.text: path for block, path in section_paths(extract("notes.html", html).blocks)}
    assert paths["A bug was fixed."] == ("Release", "Fixes")


def test_a_deeper_heading_does_not_stay_in_the_path():
    blocks = [Block("A", heading_level=1, kind="heading"), Block("under a"),
              Block("B", heading_level=2, kind="heading"), Block("under b"),
              Block("C", heading_level=1, kind="heading"), Block("under c")]
    paths = {block.text: path for block, path in section_paths(blocks)}
    assert paths["under b"] == ("A", "B")
    assert paths["under c"] == ("C",)


def test_csv_rows_are_labelled_with_their_columns():
    extraction = extract("invoices.csv", "id,amount\n7,1500\n".encode())
    assert extraction.blocks[0].kind == "table"
    assert "id: 7 | amount: 1500" in extraction.blocks[0].text


def test_a_common_section_is_what_a_packed_chunk_is_labelled_with():
    assert common_path([("Policy", "Leave"), ("Policy", "Remote")]) == ("Policy",)
    assert common_path([("Policy", "Leave"), ("Policy", "Leave")]) == ("Policy", "Leave")
    assert common_path([("A",), ("B",)]) == ()


# --- chunking ---


def test_token_count_is_larger_for_persian_than_for_latin_of_the_same_length():
    latin = "a" * 60
    persian = "ک" * 60
    assert count_tokens(persian) > count_tokens(latin) > 0
    assert count_tokens("") == 0


def test_sentences_split_on_persian_punctuation():
    assert sentences("اولی است. دومی است؟ سومی!") == ["اولی است.", "دومی است؟", "سومی!"]


def test_a_chunk_crosses_a_page_break_so_the_fact_across_it_survives():
    """The evaluation PDF's cold-chain explanation starts on page 1 and ends on page 2."""
    extraction = extract("annual-review.en.pdf", build_pdf([PAGE_ONE, PAGE_TWO]))
    pieces, _ = build_pieces(extraction.blocks, 250, 40, 900)
    spanning = [piece for piece in pieces
                if "blocked condenser coil" in piece.content and "inspection severity" in piece.content]
    assert spanning, [piece.content[-80:] for piece in pieces]
    assert spanning[0].page_start == 1 and spanning[0].page_end == 2


def test_small_sections_are_packed_instead_of_becoming_tiny_chunks():
    markdown = "# Handbook\n" + "".join(
        f"\n## Section {n}\n\nA short rule number {n} that stands alone.\n" for n in range(6)
    )
    pieces, _ = build_pieces(extract("handbook.md", markdown.encode()).blocks, 250, 40, 900)
    assert len(pieces) < 6
    # Packed across sections, the chunk is labelled with the heading they share.
    assert pieces[0].section_path == "Handbook"


def test_a_large_section_is_split_and_the_parts_overlap():
    body = " ".join(f"Sentence number {n} carries a fact." for n in range(120))
    pieces, _ = build_pieces([Block(text=body)], 120, 40, 900)
    assert len(pieces) > 2
    for earlier, later in zip(pieces, pieces[1:]):
        tail = earlier.content.split(".")[-2].strip()
        assert tail and tail in later.content, "consecutive chunks should overlap"


def test_every_child_is_contained_in_its_parent():
    markdown = "# Doc\n\n" + " ".join(f"Fact {n} is recorded here." for n in range(200))
    pieces, windows = build_pieces(extract("doc.md", markdown.encode()).blocks, 150, 30, 900)
    assert windows and len(windows) < len(pieces)
    for index, piece in enumerate(pieces):
        window = windows[piece.parent]
        assert index in window.children
        assert piece.content in window.content


def test_a_parent_is_larger_than_its_children_but_not_a_concatenation_of_them():
    """Overlap between children must not be repeated inside the parent."""
    body = " ".join(f"Statement {n} about the system." for n in range(150))
    pieces, windows = build_pieces([Block(text=body)], 120, 40, 800)
    for window in windows:
        members = [pieces[index].content for index in window.children]
        assert len(window.content) < sum(len(text) for text in members)


def test_an_empty_document_produces_nothing():
    assert build_pieces([], 250, 40, 900) == ([], [])
    assert build_pieces([Block(text="   ")], 250, 40, 900) == ([], [])
