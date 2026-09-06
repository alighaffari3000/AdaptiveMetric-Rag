"""End-to-end golden run, guarded behind `pytest -m eval`.

It ingests the whole evaluation corpus and scores retrieval, so it is slower
than the unit tests and is excluded from the default run.

    pytest -m eval
"""

import asyncio

import pytest

from eval.harness import (
    aggregate,
    build_corpus,
    cleanup,
    load_golden,
    run_cases,
    use_scratch_data_dir,
    validate_golden,
)

pytestmark = pytest.mark.eval


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    data_dir = use_scratch_data_dir()
    from app import database

    # DATA_DIR is read when app.database is first imported, which other test
    # modules may already have done; point this run at its own file explicitly.
    original_path = database.DB_PATH
    database.DB_PATH = tmp_path_factory.mktemp("golden") / "golden.db"
    database.close_thread_connection()
    try:
        from app.main import load_settings

        database.init_db()
        settings = load_settings()
        cases = load_golden()
        asyncio.run(build_corpus(settings))
        problems, _ = validate_golden(cases)
        assert not problems, "golden set does not match the corpus: " + "; ".join(problems)
        outcomes = asyncio.run(run_cases(cases, settings))
        yield aggregate(outcomes)
    finally:
        database.close_thread_connection()
        database.DB_PATH = original_path
        cleanup(data_dir)


def test_golden_set_is_consistent_with_the_corpus(report):
    assert report["summary"]["cases"] >= 80


def test_retrieval_does_not_regress_below_the_recorded_baseline(report):
    """Guards the numbers in eval/BASELINE.md after STAIR step 4."""
    summary = report["summary"]
    assert summary["hit@5"] >= 0.82, summary
    assert summary["mrr"] >= 0.73, summary
    assert summary["ndcg@10"] >= 0.75, summary


def test_english_monolingual_slice_stays_saturated(report):
    assert report["by_tag"]["en2en"]["hit@5"] >= 0.95, report["by_tag"]["en2en"]


def test_retrieval_lands_in_the_right_section_of_a_structured_document(report):
    """The structure work is only worth keeping if this stays high."""
    summary = report["summary"]
    assert summary["section_cases"] >= 15, summary
    assert summary["section_hit@1"] >= 0.80, summary
    assert summary["section_recall@3"] >= 0.80, summary


def test_the_fixtures_structural_chunking_was_meant_to_fix_stay_fixed(report):
    """Both were zero from the first baseline until sections became chunks."""
    assert report["by_tag"]["page_boundary"]["hit@5"] == 1.0, report["by_tag"]["page_boundary"]
    assert report["by_tag"]["pdf"]["hit@5"] >= 0.95, report["by_tag"]["pdf"]


def test_a_flat_document_is_not_hurt_by_the_structure_signal(report):
    assert report["by_tag"]["structural_guard"]["hit@5"] == 1.0, report["by_tag"]["structural_guard"]


def test_the_negative_sample_is_large_enough_to_mean_something(report):
    """Eight negatives made abstention accuracy unmeasurable; thirty-eight do not."""
    assert report["summary"]["abstain_cases"] >= 38, report["summary"]
    # Deliberately low: eval/BASELINE.md records why the threshold was not
    # retuned. This guards against a silent collapse, not against the gap.
    assert report["summary"]["answerable_not_abstained"] >= 0.95, report["summary"]
