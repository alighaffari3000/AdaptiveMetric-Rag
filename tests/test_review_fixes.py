"""Regression tests for defects found in the post-implementation review."""

import asyncio

import httpx
import numpy as np
import pytest

from app import net
from app.models import QueryAnalysis
from app.retrieval import CITATION_STANDOUT_GAP, RetrievalResult, _weighted_rrf, select_grounded


# --- rank fusion: ties -------------------------------------------------------

def test_tied_values_share_a_rank_regardless_of_corpus_order():
    """Numbering ties 1, 2, 3 let insertion order decide between chunks a signal
    could not tell apart. Sparse signals produce exactly that: many 1.0s."""
    forward = _weighted_rrf({"entity": np.array([1.0, 1.0, 1.0, 0.0])}, {"entity": 1.0})
    assert forward[0] == forward[1] == forward[2]
    assert forward[3] == 0.0


def test_a_tie_group_takes_the_best_rank_in_the_group():
    fused = _weighted_rrf({"s": np.array([0.9, 0.5, 0.5, 0.1])}, {"s": 1.0})
    assert fused[0] > fused[1] == fused[2] > fused[3]
    # Two chunks tied for second both get rank 2, and the next distinct value rank 4.
    k = 60
    expected = np.array([1 / (k + 1), 1 / (k + 2), 1 / (k + 2), 1 / (k + 4)]) * (k + 1)
    assert np.allclose(fused, expected)


def test_reordering_tied_rows_does_not_change_their_fused_score():
    dense = np.array([0.7, 0.6, 0.6])
    entity = np.array([1.0, 1.0, 0.0])
    a = _weighted_rrf({"dense": dense, "entity": entity}, {"dense": .5, "entity": .5})
    b = _weighted_rrf({"dense": dense[[0, 2, 1]], "entity": entity[[0, 2, 1]]}, {"dense": .5, "entity": .5})
    assert np.allclose(a[[0, 2, 1]], b)


# --- citation floor under rank fusion ---------------------------------------

def analysis() -> QueryAnalysis:
    return QueryAnalysis(intent="exact_fact", language="fa", weights={"dense": 1.0}, entities=[],
                         numbers=[], temporal_terms=[], keywords=["قرارداد"], query_tokens=["قرارداد"])


def chunk(score: float, standout: float, dense: float = 0.7, **extra) -> dict:
    return {"id": f"c{standout}", "content": "قرارداد", "score": score, "standout": standout,
            "features": {"dense": dense, "bm25": 1.0, "entity": 0.0, "keyword": 0.0,
                         "numeric": 0.0, "temporal": 0.0, "metadata": 0.0}, **extra}


def test_rank_fusion_cites_only_chunks_close_to_the_best_similarity():
    """Fused rank scores sit near 1.0 for every chunk, so the old fraction-of-top
    floor cited all five. The gate now compares raw standout instead."""
    chunks = [chunk(1.0, 3.0), chunk(0.99, 2.0), chunk(0.98, 1.4), chunk(0.97, 0.5), chunk(0.96, -0.2)]
    result = RetrievalResult(chunks, analysis(), confidence=0.8, early_exit=False, standout=3.0, fusion="rank")
    found, cited = select_grounded(result)
    assert found
    assert [c["standout"] for c in cited] == [3.0, 2.0]
    assert all(c["standout"] >= 3.0 - CITATION_STANDOUT_GAP for c in cited)


def test_linear_fusion_keeps_the_fraction_of_top_floor():
    chunks = [chunk(0.8, 3.0), chunk(0.5, 2.0), chunk(0.2, 1.9)]
    result = RetrievalResult(chunks, analysis(), confidence=0.8, early_exit=False, standout=3.0, fusion="linear")
    _, cited = select_grounded(result)
    assert [c["score"] for c in cited] == [0.8, 0.5]


def test_a_reranked_result_uses_the_blended_score_floor_even_under_rank_fusion():
    chunks = [chunk(1.0, 3.0, rerank_score=1.0), chunk(0.3, 2.9, rerank_score=0.0)]
    result = RetrievalResult(chunks, analysis(), confidence=0.8, early_exit=False, standout=3.0, fusion="rank")
    _, cited = select_grounded(result)
    assert [c["score"] for c in cited] == [1.0], "the reranker said the second chunk does not answer"


def test_the_top_chunk_is_always_cited_when_evidence_is_found():
    result = RetrievalResult([chunk(1.0, 0.0)], analysis(), confidence=0.8, early_exit=False,
                             standout=0.0, fusion="rank")
    found, cited = select_grounded(result)
    assert found and len(cited) == 1


# --- API keys never reach error text ----------------------------------------

def test_redact_covers_prose_and_bearer_tokens_too():
    assert "hunter2" not in net.redact('quota exceeded for key=hunter2 today')
    assert "abc.def" not in net.redact('Authorization: Bearer abc.def-ghi')
    assert net.redact('the monkey=banana pair') == 'the monkey=banana pair', 'no false positive inside a word'


def test_redact_strips_query_parameter_secrets_and_leaves_the_rest():
    text = "Client error '404' for url 'https://x.test/v1beta/models/m:generateContent?key=AQ.secret-value&alt=json'"
    cleaned = net.redact(text)
    assert "secret-value" not in cleaned
    assert "?key=***&alt=json" in cleaned
    assert net.redact("no secrets here") == "no secrets here"


def test_ensure_success_raises_a_redacted_provider_error():
    request = httpx.Request("POST", "https://x.test/models/m:generateContent?key=topsecret")
    response = httpx.Response(429, request=request, text='{"error": "quota for key=topsecret"}')
    with pytest.raises(net.ProviderError) as info:
        net.ensure_success(response, "Gemini generation")
    message = str(info.value)
    assert "topsecret" not in message
    assert "HTTP 429" in message and "Gemini generation" in message


def test_ensure_success_passes_a_good_response():
    request = httpx.Request("POST", "https://x.test/ok")
    net.ensure_success(httpx.Response(200, request=request, text="{}"), "anything")


def test_generation_errors_reaching_the_chat_handler_carry_no_key(monkeypatch):
    """The answer path used raw raise_for_status(), which put the Gemini key into
    the exception the chat handler logs and falls back on."""
    from app import providers
    from app.models import AppSettings

    class FakeResponse:
        status_code = 403
        is_success = False
        text = "forbidden"

    async def fake_post(client, url, **kwargs):
        assert "key=" in str(kwargs.get("params", {})) or "generateContent" in url
        return FakeResponse()

    monkeypatch.setattr(providers, "post_with_retry", fake_post)
    settings = AppSettings(provider="gemini", model="m", api_key="leak-me")
    with pytest.raises(net.ProviderError) as info:
        asyncio.run(providers.generate(settings, "q", [{"document_name": "d", "page": 1, "content": "c"}], "fa"))
    assert "leak-me" not in str(info.value)


def test_client_facing_retrieval_error_is_redacted(fresh_db):
    from fastapi.testclient import TestClient

    from app import search as search_module
    from app.main import app

    async def failing(settings, query, filters=None):
        raise httpx.ConnectError("boom for url 'https://x.test/embed?key=leaked-key'")

    original = search_module.search
    import app.main as main_module

    main_module.search = failing
    try:
        with TestClient(app) as client:
            response = client.post("/api/chat", json={"message": "مبلغ قرارداد چقدر است؟"})
    finally:
        main_module.search = original
    assert response.status_code == 502
    assert "leaked-key" not in response.text
    assert "key=***" in response.text


# --- database ---------------------------------------------------------------

def test_fresh_database_is_created_with_the_current_schema_directly(tmp_path, monkeypatch):
    from app import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "fresh.db")
    database.close_thread_connection()
    try:
        database.init_db()
        columns = {row["name"] for row in database.rows("PRAGMA table_info(chunks)")}
        assert "vector" in columns and "embedding" not in columns
        assert database.rows("SELECT name FROM sqlite_master WHERE name='chunks_v1'") == []
    finally:
        database.close_thread_connection()


def test_closing_the_thread_connection_resets_transaction_depth(fresh_db):
    """Leftover depth from an interrupted unit of work would make the next
    outermost block look nested, so it would never commit or take the lock."""
    from app import database

    database._local.depth = 3  # what an interrupted, never-exited block leaves behind
    database.close_thread_connection()
    assert database._local.depth == 0
    with fresh_db.connect() as db:
        db.execute("INSERT INTO meta(key,value) VALUES('after','reset')")
    assert fresh_db.meta_get("after") == "reset"
