"""The structure signal: section headings voting in the retrieval mixture.

Step 1 stored a section path on every chunk. This is where it earns its keep,
and where it has to be shown not to hurt the documents that have no structure
at all - which is most of a real library.
"""

import json

import pytest

from app.models import Citation
from app.retrieval import (
    DOCUMENT_WORDS,
    _section_signal,
    analyze_query,
    apply_semantic_profile,
    retrieve,
)

INTENT_QUERIES = {
    "exact_fact": "شماره قرارداد پیمانکار",
    "temporal_fact": "تاریخ پایان قرارداد چه زمانی است؟",
    "numeric_fact": "مبلغ قرارداد چقدر است؟",
    "causal": "چرا قرارداد فسخ شد؟",
    "technical_code": "CUDA error in the api function",
    "conceptual": "این مفهوم را توضیح بده",
    "document_browse": "از متن کتاب برام بنویس",
}


def test_every_intent_profile_still_sums_to_one():
    for intent, query in INTENT_QUERIES.items():
        analysis = analyze_query(query)
        assert analysis.intent == intent, f"{query!r} routed to {analysis.intent}"
        assert abs(sum(analysis.weights.values()) - 1.0) < 1e-9, analysis.weights


def test_every_intent_gives_structure_a_share():
    for query in INTENT_QUERIES.values():
        assert analyze_query(query).weights["structure"] > 0


def test_a_heading_matters_more_to_a_conceptual_question_than_a_numeric_one():
    conceptual = analyze_query(INTENT_QUERIES["conceptual"]).weights
    numeric = analyze_query(INTENT_QUERIES["numeric_fact"]).weights
    assert conceptual["structure"] > numeric["structure"]


def test_structure_is_funded_from_metadata_and_not_from_the_dense_signal():
    """Dense weights are the phase-2 tuning; the new signal must not raid them."""
    expected_dense = {"exact_fact": .25, "temporal_fact": .16, "numeric_fact": .15,
                      "causal": .41, "technical_code": .27, "conceptual": .52,
                      "document_browse": .52}
    for intent, query in INTENT_QUERIES.items():
        assert analyze_query(query).weights["dense"] == pytest.approx(expected_dense[intent])


def test_the_signal_reads_the_section_path_and_scales_with_coverage():
    assert _section_signal(["مرخصی", "استعلاجی"], "فصل دوم > مرخصی استعلاجی") == 1.0
    assert _section_signal(["مرخصی", "استعلاجی"], "فصل دوم > مرخصی استحقاقی") == 0.5
    assert _section_signal(["مرخصی"], "فصل سوم > جبران خدمات") == 0.0


def test_the_signal_is_zero_without_a_path_or_without_terms():
    assert _section_signal(["مرخصی"], "") == 0.0
    assert _section_signal([], "فصل دوم > مرخصی") == 0.0


def test_a_word_naming_the_document_is_not_evidence_about_a_section():
    """"موضوع" means "topic"; a clause headed "موضوع قرارداد" is a coincidence."""
    assert "موضوع" in DOCUMENT_WORDS
    assert _section_signal(["موضوع", "کتاب"], "ماده ۲ — موضوع قرارداد") == 0.0
    # The same path still scores for a question that names the actual subject.
    assert _section_signal(["قرارداد"], "ماده ۲ — موضوع قرارداد") == 1.0


def test_a_query_made_only_of_document_words_scores_nothing_anywhere():
    for path in ("ماده ۲ — موضوع قرارداد", "Guide > Summary", "کتاب > متن"):
        assert _section_signal(["موضوع", "خلاصه", "book", "text"], path) == 0.0


def test_the_semantic_profile_keeps_structure_alive_and_the_total_at_one():
    adjusted = apply_semantic_profile(analyze_query(INTENT_QUERIES["conceptual"]).weights, 0.85)
    assert adjusted["dense"] == pytest.approx(0.85)
    assert adjusted["structure"] > 0
    assert sum(adjusted.values()) == pytest.approx(1.0)


def _library(db, rows):
    db.execute("INSERT INTO documents VALUES('d','guide.md','text/markdown',10,?,'2026-01-01','{}')",
               (len(rows),))
    for position, (content, path) in enumerate(rows):
        from app.retrieval import embed, tokenize

        db.execute(
            "INSERT INTO chunks(id,document_id,position,page,section,section_path,content,vector,tokens,metadata) "
            "VALUES(?,'d',?,NULL,?,?,?,?,?,'{}')",
            (f"c{position}", position, path.split(" > ")[-1] or None, path, content,
             db.encode_vector(embed(content)), json.dumps(tokenize(content))),
        )
    from app.index import index as chunk_index

    chunk_index.invalidate()


def test_the_right_heading_wins_between_two_chunks_with_the_same_body(fresh_db):
    """The whole point: identical text, and only the heading tells them apart."""
    body = "حداکثر هشت روز در سال با ارائه گواهی معتبر پذیرفته می‌شود"
    _library(fresh_db, [(body, "فصل دوم > مرخصی استحقاقی"), (body, "فصل دوم > مرخصی استعلاجی")])
    result = retrieve("سقف مرخصی استعلاجی چند روز است؟", 50, 2, fusion="rank")
    assert result.chunks
    assert "استعلاجی" in result.chunks[0]["section_path"]
    assert result.chunks[0]["features"]["structure"] > result.chunks[1]["features"]["structure"]


def test_a_library_without_any_structure_ranks_exactly_as_it_did_before(fresh_db):
    """A flat corpus must not be disturbed: the signal is flat, so RRF drops it."""
    rows = [("مبلغ کل قرارداد دو میلیارد ریال است", ""),
            ("مدت قرارداد دوازده ماه شمسی است", ""),
            ("جریمه تأخیر روزانه یک درصد است", "")]
    _library(fresh_db, rows)
    result = retrieve("مبلغ کل قرارداد چقدر است؟", 50, 3, fusion="rank")
    assert [chunk["features"]["structure"] for chunk in result.chunks] == [0.0, 0.0, 0.0]
    assert "مبلغ کل قرارداد" in result.chunks[0]["content"]


def test_a_chunk_with_no_path_is_not_punished_next_to_one_that_has_a_path(fresh_db):
    """A mixed library is normal; an unstructured file must stay reachable."""
    _library(fresh_db, [
        ("دسترس‌پذیری تضمین‌شده سامانه ۹۹.۵ درصد است", ""),
        ("متن نامرتبط درباره تحویل کالا و انبار", "فصل اول > تحویل"),
    ])
    result = retrieve("دسترس‌پذیری تضمین‌شده چند درصد است؟", 50, 2, fusion="rank")
    assert "۹۹.۵" in result.chunks[0]["content"]


def test_the_diagnostic_features_expose_the_new_signal(fresh_db):
    _library(fresh_db, [("متن بخش", "فصل اول > دورکاری")])
    result = retrieve("دورکاری", 50, 1, fusion="rank")
    assert "structure" in result.chunks[0]["features"]


def test_a_citation_can_carry_the_path_it_came_from():
    citation = Citation(id=1, document_id="d", document_name="guide.md", chunk_id="c",
                        section="مرخصی استعلاجی", section_path="فصل دوم > مرخصی استعلاجی",
                        excerpt="…", score=0.5)
    assert citation.model_dump()["section_path"] == "فصل دوم > مرخصی استعلاجی"


def test_a_citation_without_a_path_is_still_valid():
    citation = Citation(id=1, document_id="d", document_name="notes.txt", chunk_id="c",
                        excerpt="…", score=0.5)
    assert citation.section_path == ""


# --- quoting the right section of a packed chunk --------------------------

PACKED = (
    "### مرخصی استعلاجی\n"
    "حداکثر هشت روز در سال با ارائه گواهی معتبر پزشک پذیرفته می‌شود.\n"
    "### مرخصی بدون حقوق\n"
    "تا سه ماه پیوسته، پس از دو سال سابقه.\n"
    "### پاداش عملکرد\n"
    "پاداش عملکرد تا سقف دو ماه حقوق پایه است.\n"
)


@pytest.mark.parametrize("question,expected", [
    ("سقف مرخصی استعلاجی چند روز است؟", "هشت روز"),
    ("مرخصی بدون حقوق حداکثر چقدر است؟", "سه ماه"),
    ("پاداش عملکرد تا چند ماه حقوق است؟", "دو ماه حقوق پایه"),
])
def test_the_quote_comes_from_the_section_the_question_names(question, expected):
    """A chunk covering three sections must answer from the right one.

    The discriminating words often appear only in the heading: nothing in
    "تا سه ماه پیوسته، پس از دو سال سابقه" says which kind of leave it is.
    """
    from app.retrieval import best_evidence

    assert expected in best_evidence(question, PACKED)


def test_a_heading_selects_its_section_without_being_quoted_as_the_answer():
    from app.retrieval import best_evidence

    quote = best_evidence("سقف مرخصی استعلاجی چند روز است؟", PACKED)
    assert "###" not in quote
    assert quote.strip() != "### مرخصی استعلاجی"


def test_a_chunk_of_nothing_but_headings_still_yields_something_to_show():
    from app.retrieval import best_evidence

    assert best_evidence("پاداش عملکرد", "### مرخصی استعلاجی\n### پاداش عملکرد\n").strip()


def test_evidence_still_prefers_a_matching_number_in_an_unstructured_chunk():
    """The behaviour that existed before sections did must not have moved."""
    from app.retrieval import best_evidence

    source = ("قرارداد شماره ۱۳۷ در فروردین امضا شد. مبلغ کل قرارداد ۲۴۰ میلیون ریال است. "
              "قرارداد یک سال اعتبار دارد.")
    assert "۲۴۰ میلیون ریال" in best_evidence("هزینه قرارداد چقدر است؟", source,
                                              "مبلغ قرارداد ۲۴۰ میلیون ریال است [1].")


def test_a_heading_sharing_one_word_does_not_beat_a_body_holding_the_question():
    """Heading and body have to be measured on the same scale.

    Weighting a heading against a sentence score that tops out well below 1.0
    let a section whose heading shared a third of the query win over one whose
    text contained all of it.
    """
    from app.retrieval import best_evidence

    content = ("## storage policy\nEverything else is unrelated boilerplate text.\n"
               "## retention window\nFull backups remain ninety days.\n")
    assert "ninety days" in best_evidence("storage backups remain", content)


def test_a_body_line_starting_with_a_version_number_can_still_be_quoted():
    from app.retrieval import best_evidence

    content = "Release notes\n7.4.0 fixes the memory leak\nUpgrade before the end of the month.\n"
    assert "7.4.0" in best_evidence("which release fixes the memory leak", content)


def test_a_heading_with_no_text_under_it_is_not_quoted_as_an_answer():
    """A chapter heading immediately followed by a subheading has no prose.

    Such a group names a section and says nothing, so it must lose to any
    group that carries text, and be used only when there is no other.
    """
    from app.retrieval import best_evidence

    content = ("فصل اول — کلیات\nماده ۱:\nاین آیین‌نامه ترتیب انجام معاملات را تعیین می‌کند.\n"
               "ماده ۲ — تعاریف\nمعامله کوچک تا پانصد میلیون ریال است.\n")
    quote = best_evidence("مرخصی بدون حقوق حداکثر چقدر است؟", content)
    assert quote.strip() != "فصل اول — کلیات"
    assert best_evidence("کلیات", "فصل اول — کلیات\nفصل دوم — روش\n").strip(), \
        "a chunk of nothing but headings still has to show something"
