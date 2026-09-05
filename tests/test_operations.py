"""Phase 7: secrets at rest, access control, logging, and one full journey."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import secrets
from app.observability import JsonFormatter, request_id
from app.security import reset_limits

from tests.test_api_smoke import CONTRACT, RUNBOOK, upload


@pytest.fixture
def client(fresh_db):
    from app.main import app

    reset_limits()
    with TestClient(app) as test_client:
        yield test_client
    reset_limits()


@pytest.fixture
def secret_key(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "a passphrase from the operator's secret store")
    yield


# --- API keys at rest ---


def test_a_key_is_unreadable_in_the_database(secret_key):
    stored = secrets.encrypt("sk-real-key-value")
    assert stored.startswith(secrets.PREFIX)
    assert "sk-real-key-value" not in stored
    assert secrets.decrypt(stored) == "sk-real-key-value"


def test_without_a_secret_the_key_is_stored_as_before(monkeypatch):
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    assert secrets.encrypt("sk-plain") == "sk-plain"
    assert secrets.decrypt("sk-plain") == "sk-plain"


def test_a_rotated_secret_costs_the_key_and_not_the_application(secret_key, monkeypatch):
    stored = secrets.encrypt("sk-real-key-value")
    monkeypatch.setenv("APP_SECRET_KEY", "a different passphrase")
    assert secrets.decrypt(stored) == ""


def test_saving_a_key_encrypts_it_and_never_returns_it(client, secret_key):
    settings = client.get("/api/settings").json()
    settings["api_key"] = "sk-secret-value"
    saved = client.put("/api/settings", json=settings)
    assert saved.status_code == 200, saved.text
    assert saved.json()["api_key"] == "" and saved.json()["has_api_key"] is True

    from app import database

    row = database.row("SELECT value FROM settings WHERE id=1")
    assert "sk-secret-value" not in row["value"]
    from app.main import load_settings

    assert load_settings().api_key == "sk-secret-value"


def test_production_refuses_to_store_a_key_at_all(client, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    settings = client.get("/api/settings").json()
    settings["api_key"] = "sk-should-not-be-stored"
    rejected = client.put("/api/settings", json=settings)
    assert rejected.status_code == 403
    assert "environment" in rejected.json()["detail"]


def test_the_interface_is_told_how_keys_are_handled(client, secret_key):
    body = client.get("/api/system/info").json()
    assert body["secrets"] == {"production": False, "encryption": True}


# --- access control ---


def test_no_token_configured_means_no_token_required(client, monkeypatch):
    monkeypatch.delenv("APP_AUTH_TOKEN", raising=False)
    assert client.get("/api/documents").status_code == 200


def test_a_configured_token_is_required(client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "letmein")
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/documents", headers={"Authorization": "Bearer letmein"}).status_code == 200


def test_health_stays_reachable_for_a_container_probe(client, monkeypatch):
    monkeypatch.setenv("APP_AUTH_TOKEN", "letmein")
    assert client.get("/health").status_code == 200


def test_the_expensive_endpoints_are_rate_limited(client, monkeypatch):
    monkeypatch.setenv("APP_RATE_LIMIT_CHAT", "3")
    reset_limits()
    codes = [client.post("/api/chat", json={"message": "مبلغ قرارداد چقدر است؟"}).status_code
             for _ in range(4)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_reading_is_never_rate_limited(client, monkeypatch):
    monkeypatch.setenv("APP_RATE_LIMIT_CHAT", "1")
    reset_limits()
    assert all(client.get("/api/documents").status_code == 200 for _ in range(5))


def test_the_limit_can_be_switched_off(client, monkeypatch):
    monkeypatch.setenv("APP_RATE_LIMIT_CHAT", "0")
    reset_limits()
    codes = [client.post("/api/chat", json={"message": "مبلغ قرارداد چقدر است؟"}).status_code
             for _ in range(4)]
    assert codes == [200, 200, 200, 200]


# --- logging ---


def test_a_log_record_is_one_json_object_carrying_the_request_id():
    token = request_id.set("abc123")
    record = logging.LogRecord("adaptive_metric_rag.test", logging.INFO, __file__, 1,
                               "retrieved %d chunks", (5,), None)
    record.duration_ms = 12.5
    payload = json.loads(JsonFormatter().format(record))
    request_id.reset(token)
    assert payload["message"] == "retrieved 5 chunks"
    assert payload["request_id"] == "abc123"
    assert payload["duration_ms"] == 12.5
    assert payload["level"] == "info"


def test_the_response_carries_the_request_id_back(client):
    response = client.get("/health")
    assert response.headers["X-Request-ID"]


def test_a_caller_supplied_request_id_is_kept(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-42"})
    assert response.headers["X-Request-ID"] == "trace-42"


# --- the original file ---


def test_the_uploaded_file_can_be_read_back(client):
    document = upload(client, "contract-137.fa.txt", CONTRACT)
    response = client.get(f"/api/documents/{document['id']}/file")
    assert response.status_code == 200
    assert response.content == CONTRACT


def test_deleting_a_document_removes_its_file(client):
    document = upload(client, "contract-137.fa.txt", CONTRACT)
    client.delete(f"/api/documents/{document['id']}")
    assert client.get(f"/api/documents/{document['id']}/file").status_code == 404


def test_a_crafted_file_name_cannot_escape_the_upload_directory():
    from app.documents import original_path

    assert original_path("../../etc/passwd") is None
    assert original_path("") is None
    assert original_path(".env") is None


# --- the whole journey ---


def test_upload_chat_cite_open_and_delete(client):
    """One document's life: in, answered from, cited, opened, and gone."""
    contract = upload(client, "contract-137.fa.txt", CONTRACT)
    upload(client, "gpu-runbook.en.md", RUNBOOK)
    assert client.get("/health").json()["documents"] == 2

    answer = client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()
    assert answer["evidence_found"] is True
    citation = answer["citations"][0]
    assert citation["document_id"] == contract["id"]
    assert "۲٬۴۰۰٬۰۰۰٬۰۰۰" in citation["excerpt"]

    # the cited chunk is readable, and so is the document it came from
    assert client.get(f"/api/chunks/{citation['chunk_id']}").json()["id"] == citation["chunk_id"]
    assert client.get(f"/api/documents/{contract['id']}/file").content == CONTRACT

    # the follow-up is answered from the same conversation
    follow_up = client.post("/api/chat", json={"message": "و کی تموم می‌شه؟",
                                               "conversation_id": answer["conversation_id"]}).json()
    assert follow_up["citations"][0]["document_id"] == contract["id"]

    export = client.get(f"/api/conversations/{answer['conversation_id']}/export").json()
    assert len(export["messages"]) == 4
    assert "api_key" not in json.dumps(export["runtime"])

    assert client.delete(f"/api/documents/{contract['id']}").status_code == 200
    after = client.post("/api/chat", json={"message": "مبلغ کل قرارداد چقدر است؟"}).json()
    assert after["evidence_found"] is False
    assert client.get("/health").json()["documents"] == 1
