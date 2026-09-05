"""Tests for embedding provider selection, query handling, and deduplication."""

import asyncio
import json

import httpx
import pytest

from app import embeddings, net
from app.documents import content_digest, find_duplicate
from app.models import AppSettings


def settings_for(provider: str, **overrides) -> AppSettings:
    return AppSettings(embedding_provider=provider, **overrides)


def test_feature_hashing_is_reported_as_non_semantic():
    info = embeddings.embedding_info(settings_for("local"))
    assert info["semantic"] is False
    assert info["dimensions"] == 384
    assert info["requires_api_key"] is False


@pytest.mark.parametrize("provider", ["ollama", "sentence-transformers", "openai", "gemini"])
def test_every_model_backed_provider_is_reported_as_semantic(provider):
    assert embeddings.embedding_info(settings_for(provider))["semantic"] is True


def test_hosted_providers_declare_that_they_need_a_key():
    assert embeddings.embedding_info(settings_for("openai"))["requires_api_key"] is True
    assert embeddings.embedding_info(settings_for("gemini"))["requires_api_key"] is True
    assert embeddings.embedding_info(settings_for("ollama"))["requires_api_key"] is False


def test_e5_models_get_the_prefixes_they_were_trained_with():
    assert embeddings._prefix("intfloat/multilingual-e5-small", "query") == "query: "
    assert embeddings._prefix("intfloat/multilingual-e5-small", "document") == "passage: "
    assert embeddings._prefix("BAAI/bge-m3", "query") == ""


def test_embedding_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert embeddings._embedding_key(settings_for("gemini")) == "from-env"
    explicit = settings_for("gemini", embedding_api_key="explicit")
    assert embeddings._embedding_key(explicit) == "explicit"


def test_embedding_key_is_shared_only_when_both_providers_match(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    shared = AppSettings(provider="gemini", api_key="shared", embedding_provider="gemini")
    assert embeddings._embedding_key(shared) == "shared"
    separate = AppSettings(provider="openai", api_key="generation-only", embedding_provider="gemini")
    assert embeddings._embedding_key(separate) == ""


def test_semantic_query_is_embedded_as_written(monkeypatch):
    seen: list[list[str]] = []

    async def fake(settings, texts, kind="document"):
        seen.append(texts)
        return [[1.0, 0.0]]

    monkeypatch.setattr(embeddings, "create_embeddings", fake)
    question = "قرارداد شماره ۱۳۷ چه زمانی تمام می‌شود؟"
    asyncio.run(embeddings.create_query_embedding(settings_for("gemini"), question))
    assert seen == [[question]], "a semantic model must not be fed blended keyword variants"


def test_feature_hashing_query_still_blends_keyword_variants(monkeypatch):
    seen: list[list[str]] = []

    async def fake(settings, texts, kind="document"):
        seen.append(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(embeddings, "create_embeddings", fake)
    asyncio.run(embeddings.create_query_embedding(settings_for("local"), "موضوع کتاب چیست؟"))
    assert len(seen[0]) > 1, "feature hashing needs the cross-language variants to match anything"


def test_local_provider_needs_no_network():
    vectors = asyncio.run(embeddings.create_embeddings(settings_for("local"), ["قرارداد", "contract"]))
    assert len(vectors) == 2 and len(vectors[0]) == 384


def test_unknown_provider_is_rejected():
    broken = settings_for("local").model_copy(update={"embedding_provider": "nope"})
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        asyncio.run(embeddings.create_embeddings(broken, ["text"]))


def test_missing_sentence_transformers_explains_the_alternative(monkeypatch):
    monkeypatch.setitem(embeddings._st_models, "x", None)
    embeddings._st_models.pop("x")
    monkeypatch.setattr(embeddings, "_st_models", {})

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ValueError, match="pip install"):
        embeddings._load_sentence_transformer("intfloat/multilingual-e5-small")


def test_gemini_request_uses_retrieval_task_types_and_normalises(monkeypatch):
    captured: list[dict] = []

    class FakeResponse:
        status_code = 200
        is_success = True
        text = ""

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    async def fake_post(client, url, **kwargs):
        captured.append({"url": url, "json": kwargs["json"]})
        count = len(kwargs["json"]["requests"])
        return FakeResponse({"embeddings": [{"values": [3.0, 4.0]} for _ in range(count)]})

    monkeypatch.setattr(embeddings, "post_with_retry", fake_post)
    settings = settings_for("gemini", embedding_model="gemini-embedding-001", embedding_api_key="k")

    document_vectors = asyncio.run(embeddings.create_embeddings(settings, ["متن سند"], kind="document"))
    query_vectors = asyncio.run(embeddings.create_embeddings(settings, ["سؤال"], kind="query"))

    assert captured[0]["json"]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert captured[1]["json"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert "models/gemini-embedding-001:batchEmbedContents" in captured[0]["url"]
    # A truncated Gemini vector is not unit length; it must be normalised before storage.
    assert document_vectors == [[0.6, 0.8]] and query_vectors == [[0.6, 0.8]]


def test_gemini_rejects_a_short_response(monkeypatch):
    class FakeResponse:
        status_code = 200
        is_success = True
        text = ""

        def json(self):
            return {"embeddings": [{"values": [1.0]}]}

    async def fake_post(client, url, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(embeddings, "post_with_retry", fake_post)
    settings = settings_for("gemini", embedding_api_key="k")
    with pytest.raises(ValueError, match="unexpected number"):
        asyncio.run(embeddings.create_embeddings(settings, ["a", "b"]))


def test_hosted_providers_fail_clearly_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        asyncio.run(embeddings.create_embeddings(settings_for("gemini"), ["text"]))
    with pytest.raises(ValueError, match="API key"):
        asyncio.run(embeddings.create_embeddings(settings_for("openai"), ["text"]))


def test_retry_gives_up_after_the_last_attempt(monkeypatch):
    attempts = {"count": 0}

    class FailingClient:
        async def post(self, url, **kwargs):
            attempts["count"] += 1
            raise httpx.ConnectTimeout("no route")

    monkeypatch.setattr(net, "RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(net, "RETRY_MAX_DELAY", 0)
    with pytest.raises(httpx.ConnectTimeout):
        asyncio.run(net.post_with_retry(FailingClient(), "http://example.invalid"))
    assert attempts["count"] == net.RETRY_ATTEMPTS


def test_retry_recovers_from_a_transient_failure(monkeypatch):
    class FlakyClient:
        def __init__(self):
            self.calls = 0

        async def post(self, url, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("slow")

            class Ok:
                status_code = 200

            return Ok()

    monkeypatch.setattr(net, "RETRY_BASE_DELAY", 0)
    monkeypatch.setattr(net, "RETRY_MAX_DELAY", 0)
    client = FlakyClient()
    response = asyncio.run(net.post_with_retry(client, "http://example.invalid"))
    assert response.status_code == 200 and client.calls == 2


def test_identical_uploads_are_detected_by_content(fresh_db):
    payload = "قرارداد شماره ۱۳۷".encode("utf-8")
    digest = content_digest(payload)
    assert find_duplicate(digest) is None

    fresh_db.execute(
        "INSERT INTO documents(id,name,type,size,chunks,created_at,metadata) VALUES(?,?,?,?,?,?,?)",
        ("d1", "contract.txt", "text/plain", len(payload), 1, "2026-01-01",
         json.dumps({"sha256": digest})),
    )
    found = find_duplicate(digest)
    assert found and found["name"] == "contract.txt"
    assert find_duplicate(content_digest(b"different content")) is None
