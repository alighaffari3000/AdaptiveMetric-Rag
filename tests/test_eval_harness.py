"""Unit tests for the evaluation harness itself.

These run without a database: they check the yardstick, not the retriever.
The full golden run lives behind `pytest -m eval`.
"""

from eval.harness import GoldenCase, load_golden, score_ranking
from eval.normalize import contains, normalize


def chunk(content: str, document_name: str = "doc.txt") -> dict:
    return {"content": content, "document_name": document_name}


def test_normalize_folds_persian_spelling_variants():
    assert normalize("ك") == normalize("ک")
    assert normalize("ي") == normalize("ی")
    assert normalize("۱۴۰۲") == "1402"
    assert normalize("مي‌شود") == normalize("می شود")
    assert normalize("۲٬۴۰۰٬۰۰۰") == "2400000"


def test_contains_matches_across_digit_and_letter_variants():
    assert contains("مبلغ ۲٬۴۰۰٬۰۰۰٬۰۰۰ ریال است", "2400000000 ریال")
    assert not contains("مبلغ ۲۹۰٬۰۰۰٬۰۰۰ ریال است", "2400000000")


def test_case_requires_every_must_contain_term():
    case = GoldenCase(id="c", question="q", language="fa", doc="doc.txt",
                      must_contain=["شرکت آفتاب", "شرکت سپهر"])
    assert case.matches(chunk("قرارداد میان شرکت آفتاب و شرکت سپهر امضا شد"))
    assert not case.matches(chunk("قرارداد میان شرکت آفتاب و شرکت مهرگان امضا شد"))


def test_case_rejects_right_content_in_the_wrong_document():
    case = GoldenCase(id="c", question="q", language="en", doc="a.txt", must_contain=["EUR 186,000"])
    assert case.matches(chunk("The fee is EUR 186,000 per year", "a.txt"))
    assert not case.matches(chunk("The fee is EUR 186,000 per year", "b.txt"))


def test_any_contain_needs_only_one_term():
    case = GoldenCase(id="c", question="q", language="fa", doc="doc.txt",
                      any_contain=["عادت اول", "عادت دوم"])
    assert case.matches(chunk("عادت دوم، تفکیک تولید از قضاوت است"))
    assert not case.matches(chunk("عادت سوم، محدودیت عمدی است"))


def test_score_ranking_rewards_an_earlier_first_hit():
    case = GoldenCase(id="c", question="q", language="en", doc="d.txt", must_contain=["target"])
    early = score_ranking(case, [chunk("target here", "d.txt"), chunk("noise", "d.txt")], total_relevant=1)
    late = score_ranking(case, [chunk("noise", "d.txt"), chunk("target here", "d.txt")], total_relevant=1)
    assert early["mrr"] == 1.0
    assert late["mrr"] == 0.5
    assert early["ndcg@10"] > late["ndcg@10"]
    assert early["hit@5"] == late["hit@5"] == 1.0


def test_score_ranking_reports_a_miss_as_zero():
    case = GoldenCase(id="c", question="q", language="en", doc="d.txt", must_contain=["target"])
    metrics = score_ranking(case, [chunk("noise", "d.txt")], total_relevant=1)
    assert metrics == {"hit@5": 0.0, "recall@5": 0.0, "mrr": 0.0, "ndcg@10": 0.0, "first_rank": 0.0}


def test_recall_counts_every_relevant_chunk_in_the_top_five():
    case = GoldenCase(id="c", question="q", language="en", doc="d.txt", must_contain=["target"])
    ranked = [chunk("target a", "d.txt"), chunk("noise", "d.txt"), chunk("target b", "d.txt")]
    assert score_ranking(case, ranked, total_relevant=4)["recall@5"] == 0.5


def test_golden_file_loads_and_declares_expectations():
    cases = load_golden()
    assert len(cases) >= 80
    assert {case.id for case in cases}.__len__() == len(cases)
    for case in cases:
        assert case.question.strip(), f"{case.id} has an empty question"
        assert case.language in {"fa", "en"}, f"{case.id} has an odd language"
        assert case.tags, f"{case.id} carries no tag"
        if case.expect_abstain:
            assert not case.doc and not case.must_contain and not case.any_contain
        else:
            assert case.doc, f"{case.id} names no target document"
            assert case.must_contain or case.any_contain, f"{case.id} states no expected content"


def test_golden_file_covers_every_weak_area_the_baseline_calls_out():
    tags = {tag for case in load_golden() for tag in case.tags}
    assert {"cross_lingual", "fa2en", "en2fa", "follow_up", "page_boundary",
            "abstain", "multi_intent", "pdf"} <= tags


def test_unknown_golden_fields_are_rejected():
    try:
        GoldenCase.from_json({"id": "x", "question": "q", "language": "fa", "typo_field": 1})
    except ValueError as exc:
        assert "typo_field" in str(exc)
    else:
        raise AssertionError("unknown fields must not be silently accepted")


def sectioned(content: str, section_path: str, document_name: str = "doc.txt") -> dict:
    return {"content": content, "document_name": document_name, "section_path": section_path}


def test_relevant_section_is_an_accepted_field():
    case = GoldenCase.from_json({"id": "s", "question": "q", "language": "fa", "doc": "d.md",
                                 "must_contain": ["x"], "relevant_section": "فصل دوم > مرخصی"})
    assert case.relevant_section == "فصل دوم > مرخصی"


def test_section_match_reads_the_path_and_folds_spelling():
    case = GoldenCase(id="s", question="q", language="fa", doc="d.md",
                      relevant_section="مرخصی استعلاجی")
    assert case.section_matches(sectioned("متن", "فصل دوم > مرخصی استعلاجی", "d.md"))
    # Arabic yeh in the corpus path must still match the Persian yeh in the case.
    assert case.section_matches(sectioned("متن", "فصل دوم > مرخصي استعلاجي", "d.md"))
    assert not case.section_matches(sectioned("متن", "فصل دوم > مرخصی استحقاقی", "d.md"))


def test_section_match_rejects_the_right_section_of_the_wrong_document():
    case = GoldenCase(id="s", question="q", language="fa", doc="a.md", relevant_section="دورکاری")
    assert case.section_matches(sectioned("متن", "فصل اول > دورکاری", "a.md"))
    assert not case.section_matches(sectioned("متن", "فصل اول > دورکاری", "b.md"))


def test_a_case_without_a_declared_section_never_matches_one():
    case = GoldenCase(id="s", question="q", language="fa", doc="d.md", must_contain=["x"])
    assert not case.section_matches(sectioned("x", "فصل اول", "d.md"))


def test_section_match_falls_back_to_the_single_level_section_column():
    case = GoldenCase(id="s", question="q", language="fa", doc="d.docx", relevant_section="Rollback")
    assert case.section_matches({"content": "c", "document_name": "d.docx", "section": "Rollback"})


def test_section_metrics_are_absent_for_cases_that_declare_no_section():
    case = GoldenCase(id="s", question="q", language="en", doc="d.txt", must_contain=["target"])
    metrics = score_ranking(case, [chunk("target", "d.txt")], total_relevant=1)
    assert "section_hit@1" not in metrics
    assert "section_recall@3" not in metrics


def test_section_hit_at_one_only_counts_the_top_chunk():
    case = GoldenCase(id="s", question="q", language="fa", doc="d.md",
                      must_contain=["پاسخ"], relevant_section="مرخصی")
    right_first = score_ranking(case, [sectioned("پاسخ", "فصل > مرخصی", "d.md"),
                                       sectioned("پاسخ", "فصل > حقوق", "d.md")], total_relevant=1)
    right_second = score_ranking(case, [sectioned("پاسخ", "فصل > حقوق", "d.md"),
                                        sectioned("پاسخ", "فصل > مرخصی", "d.md")], total_relevant=1)
    assert right_first["section_hit@1"] == 1.0
    assert right_second["section_hit@1"] == 0.0
    assert right_first["section_recall@3"] == right_second["section_recall@3"] == 1.0


def test_section_recall_at_three_ignores_a_hit_below_the_third_rank():
    case = GoldenCase(id="s", question="q", language="fa", doc="d.md",
                      must_contain=["پاسخ"], relevant_section="مرخصی")
    ranked = [sectioned("پاسخ", "فصل > حقوق", "d.md")] * 3 + [sectioned("پاسخ", "فصل > مرخصی", "d.md")]
    assert score_ranking(case, ranked, total_relevant=1)["section_recall@3"] == 0.0


def test_golden_file_carries_structural_cases_with_sections():
    cases = load_golden()
    structural = [case for case in cases if "structural" in case.tags]
    assert len(structural) >= 15
    for case in structural:
        assert case.relevant_section, f"{case.id} is tagged structural but declares no section"
    assert any("structural_guard" in case.tags for case in cases), \
        "a flat-document guard case must exist so structure cannot silently regress it"


def test_abstain_cases_may_not_declare_a_section():
    for case in load_golden():
        if case.expect_abstain:
            assert not case.relevant_section, f"{case.id} abstains but names a section"
