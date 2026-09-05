"""End-to-end smoke test over the HTTP surface.

Phase 1 rewrote the storage layer, the retrieval path, and application startup.
This exercises upload, chat, citations, filtering, and delete against the real
app so that refactor cannot quietly break the API.
"""

import pytest
from fastapi.testclient import TestClient

from app import database


CONTRACT = """قرارداد شماره ۱۳۷ میان شرکت آفتاب و شرکت سپهر در تاریخ ۱۴۰۵/۰۱/۱۵ امضا شد.
مبلغ کل قرارداد ۲٬۴۰۰٬۰۰۰٬۰۰۰ ریال است و در چهار قسط مساوی پرداخت می‌شود.
قرارداد در تاریخ ۱۴۰۶/۰۱/۱۴ خاتمه می‌یابد مگر آن‌که تمدید کتبی امضا شود.
""".encode("utf-8")

RUNBOOK = b"""GPU training runbook.
A CUDA out of memory error is usually caused by an oversized batch, not a leak.
Reduce per_device_train_batch_size to 8 and enable gradient checkpointing.
"""


@pytest.fixture
def client(fresh_db):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def upload(client, name: str, payload: bytes):
    response = client.post("/api/documents", files={"file": (name, payload, "text/plain")})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_an_empty_corpus(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 0 and body["chunks"] == 0


def test_upload_then_chat_returns_grounded_citations(client):
    document = upload(client, "contract-137.fa.txt", CONTRACT)
    assert document["chunks"] >= 1

    body = client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()
    assert body["evidence_found"] is True
    assert body["citations"], body
    citation = body["citations"][0]
    assert citation["document_name"] == "contract-137.fa.txt"
    assert citation["chunk_id"]
    assert "۲٬۴۰۰٬۰۰۰٬۰۰۰" in citation["excerpt"]
    assert body["analysis"]["language"] == "fa"
    assert client.get(f"/api/chunks/{citation['chunk_id']}").status_code == 200


def test_index_picks_up_a_second_document_without_a_restart(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    upload(client, "gpu-runbook.en.md", RUNBOOK)

    body = client.post("/api/chat", json={"message": "How do I fix CUDA out of memory?"}).json()
    assert body["citations"][0]["document_name"] == "gpu-runbook.en.md"
    assert client.get("/health").json()["documents"] == 2


def test_document_filter_restricts_retrieval(client):
    contract = upload(client, "contract-137.fa.txt", CONTRACT)
    upload(client, "gpu-runbook.en.md", RUNBOOK)

    body = client.post("/api/chat", json={
        "message": "CUDA out of memory",
        "filters": {"document_id": contract["id"]},
    }).json()
    for citation in body["citations"]:
        assert citation["document_id"] == contract["id"]


def test_chat_abstains_when_the_corpus_cannot_answer(client):
    upload(client, "gpu-runbook.en.md", RUNBOOK)
    body = client.post("/api/chat", json={"message": "قرارداد شماره ۱۳۷ چه زمانی تمام می‌شود؟"}).json()
    assert body["citations"] == [] or body["evidence_found"] is False


def test_deleting_a_document_removes_it_from_retrieval(client):
    document = upload(client, "contract-137.fa.txt", CONTRACT)
    assert client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()["citations"]

    assert client.delete(f"/api/documents/{document['id']}").status_code == 200
    assert client.get("/health").json()["chunks"] == 0
    body = client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()
    assert body["citations"] == []
    assert body["evidence_found"] is False


def test_system_info_reports_index_state(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    body = client.get("/api/system/info").json()
    assert body["embedding"]["dimensions"] == 384
    assert body["index"]["chunks"] >= 1
    assert body["index"]["stale_vectors"] == 0


def test_conversation_is_persisted_and_exportable(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    first = client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()
    conversation_id = first["conversation_id"]
    client.post("/api/chat", json={"message": "تاریخ امضا چه بود؟", "conversation_id": conversation_id})

    messages = client.get(f"/api/conversations/{conversation_id}").json()
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]

    export = client.get(f"/api/conversations/{conversation_id}/export").json()
    assert export["format"] == "adaptivemetric-rag-conversation"
    assert "api_key" not in export["runtime"]
    assert len(export["messages"]) == 4


def test_citation_highlight_migration_runs_only_once(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"})
    stamp = database.meta_get("citation_highlights_refreshed_at")
    assert stamp, "startup should record that the one-shot migration ran"

    from app.main import app

    with TestClient(app):
        pass
    assert database.meta_get("citation_highlights_refreshed_at") == stamp


def test_a_follow_up_question_is_answered_from_the_conversation(client):
    """«و کی تموم می‌شه؟» names no contract; the previous turn does."""
    upload(client, "contract-137.fa.txt", CONTRACT)
    upload(client, "gpu-runbook.en.md", RUNBOOK)
    first = client.post("/api/chat", json={"message": "قرارداد شماره ۱۳۷ درباره چیست؟"}).json()

    follow_up = client.post("/api/chat", json={"message": "و کی تموم می‌شه؟",
                                              "conversation_id": first["conversation_id"]}).json()
    assert follow_up["evidence_found"] is True
    assert follow_up["citations"][0]["document_name"] == "contract-137.fa.txt"
    assert follow_up["analysis"]["rewritten_from"] == "و کی تموم می‌شه؟"
    assert follow_up["analysis"]["rewrite_source"] == "rules"


def test_a_new_conversation_carries_no_context(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    client.post("/api/chat", json={"message": "قرارداد شماره ۱۳۷ درباره چیست؟"})
    fresh = client.post("/api/chat", json={"message": "و کی تموم می‌شه؟"}).json()
    assert fresh["analysis"]["rewrite_source"] == ""


def test_stored_tokens_are_re_derived_once_after_a_normalizer_change(client):
    upload(client, "contract-137.fa.txt", CONTRACT)
    assert database.meta_get("text_normalizer_version")

    # A library written by an older normalizer: Persian digits left as they were.
    database.execute("UPDATE chunks SET tokens=?", ('["قرارداد","۱۳۷"]',))
    database.execute("DELETE FROM meta WHERE key='text_normalizer_version'")

    from app.main import app

    with TestClient(app) as restarted:
        body = restarted.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()
    assert body["citations"], "re-derived tokens should make the chunk findable again"
    stored = database.row("SELECT tokens FROM chunks")["tokens"]
    assert "137" in stored and "۱۳۷" not in stored


def test_unsupported_upload_is_rejected(client):
    response = client.post("/api/documents", files={"file": ("notes.xyz", b"data", "text/plain")})
    assert response.status_code == 415
