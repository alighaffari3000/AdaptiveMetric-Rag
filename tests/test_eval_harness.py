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
