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
    summary = report["summary"]
    assert summary["hit@5"] >= 0.78, summary
    assert summary["mrr"] >= 0.69, summary
    assert summary["ndcg@10"] >= 0.71, summary


def test_english_monolingual_slice_stays_saturated(report):
    assert report["by_tag"]["en2en"]["hit@5"] >= 0.95, report["by_tag"]["en2en"]
