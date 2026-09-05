from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from typing import Any, Literal

import httpx

from . import database
from .index import index
from .net import ensure_success, post_with_retry
from .models import AppSettings
from .retrieval import DIMENSION, embed as local_embed, query_variants


logger = logging.getLogger("adaptive_metric_rag.embeddings")

TextKind = Literal["document", "query"]

# The no-model default is feature hashing over tokens. It is fast and offline but
# it is a lexical signal, not a semantic one: a Persian question and its English
# answer share no features at all. Anything else in this table is semantic.
NON_SEMANTIC_PROVIDERS = {"local"}

_st_models: dict[str, Any] = {}
_st_lock = threading.Lock()

DEFAULT_OLLAMA_URL = "http://host.docker.internal:11434"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_ST_MODEL = "intfloat/multilingual-e5-small"
GEMINI_DIMENSIONS = 768
BATCH_SIZE = 32


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [float(value) / norm for value in vector]


def is_semantic(settings: AppSettings) -> bool:
    return settings.embedding_provider not in NON_SEMANTIC_PROVIDERS


def _embedding_key(settings: AppSettings) -> str:
    if settings.embedding_api_key:
        return settings.embedding_api_key
    if settings.provider == settings.embedding_provider and settings.api_key:
        return settings.api_key
    if settings.embedding_provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    if settings.embedding_provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    return ""


def _base_url(settings: AppSettings) -> str:
    if settings.embedding_base_url:
        return settings.embedding_base_url.rstrip("/")
    defaults = {
        "ollama": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "openai": DEFAULT_OPENAI_URL,
        "gemini": DEFAULT_GEMINI_URL,
    }
    return defaults.get(settings.embedding_provider, "").rstrip("/")


def _prefix(model: str, kind: TextKind) -> str:
    """The e5 family is trained with explicit query/passage markers."""
    if "e5" in model.lower():
        return "query: " if kind == "query" else "passage: "
    return ""


def embedding_info(settings: AppSettings) -> dict:
    provider = settings.embedding_provider
    common = {
        "provider": provider,
        "model": settings.embedding_model,
        "semantic": is_semantic(settings),
        "base_url": _base_url(settings),
        "requires_api_key": provider in {"openai", "gemini"},
        "has_api_key": bool(_embedding_key(settings)),
        "dimensions": None,
    }
    if provider == "local":
        return {**common,
                "model": "multilingual-feature-hashing-v1",
                "display_name": "هش ویژگی محلی (سریع، بدون مدل، غیرمعنایی)",
                "dimensions": DIMENSION}
    display = {
        "ollama": f"{settings.embedding_model} (Ollama)",
        "sentence-transformers": f"{settings.embedding_model} (محلی)",
        "openai": f"{settings.embedding_model} (OpenAI-compatible)",
        "gemini": f"{settings.embedding_model} (Google Gemini)",
    }
    return {**common, "display_name": display.get(provider, settings.embedding_model)}


def _load_sentence_transformer(model_name: str):
    with _st_lock:
        model = _st_models.get(model_name)
        if model is not None:
            return model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the install profile
            raise ValueError(
                "The sentence-transformers embedding provider needs the optional extra: "
                "pip install 'sentence-transformers>=3,<6'. Use the Ollama, OpenAI or "
                "Gemini provider to embed without a local model."
            ) from exc
        logger.info("loading local embedding model %s (first use downloads it)", model_name)
        model = SentenceTransformer(model_name)
        _st_models[model_name] = model
        return model


def _embed_sentence_transformers(settings: AppSettings, texts: list[str], kind: TextKind) -> list[list[float]]:
    model_name = settings.embedding_model or DEFAULT_ST_MODEL
    model = _load_sentence_transformer(model_name)
    prefix = _prefix(model_name, kind)
    prepared = [prefix + text for text in texts]
    vectors = model.encode(prepared, batch_size=BATCH_SIZE, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    return [vector.tolist() for vector in vectors]


async def _embed_ollama(settings: AppSettings, texts: list[str], client: httpx.AsyncClient) -> list[list[float]]:
    base_url = _base_url(settings) or DEFAULT_OLLAMA_URL
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = await post_with_retry(client, f"{base_url}/api/embed",
                                          json={"model": settings.embedding_model, "input": batch})
        if response.status_code == 404:
            # Compatibility with older Ollama releases.
            for text in batch:
                legacy = await post_with_retry(client, f"{base_url}/api/embeddings",
                                                json={"model": settings.embedding_model, "prompt": text})
                ensure_success(legacy, "Ollama embedding")
                vectors.append(_normalize(legacy.json()["embedding"]))
            continue
        ensure_success(response, "Ollama embedding")
        payload = response.json()
        batch_vectors = payload.get("embeddings") or ([payload["embedding"]] if payload.get("embedding") else [])
        if len(batch_vectors) != len(batch):
            raise ValueError("Ollama returned an unexpected number of embeddings")
        vectors.extend(_normalize(vector) for vector in batch_vectors)
    return vectors


async def _embed_openai(settings: AppSettings, texts: list[str], client: httpx.AsyncClient) -> list[list[float]]:
    base_url = _base_url(settings) or DEFAULT_OPENAI_URL
    api_key = _embedding_key(settings)
    if not api_key:
        raise ValueError("An API key is required for the OpenAI-compatible embedding provider")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = await post_with_retry(
            client,
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": settings.embedding_model, "input": batch},
        )
        ensure_success(response, "OpenAI embedding")
        data = response.json()["data"]
        if len(data) != len(batch):
            raise ValueError("The embedding endpoint returned an unexpected number of vectors")
        vectors.extend(_normalize(item["embedding"]) for item in sorted(data, key=lambda d: d.get("index", 0)))
    return vectors


async def _embed_gemini(settings: AppSettings, texts: list[str], kind: TextKind,
                        client: httpx.AsyncClient) -> list[list[float]]:
    base_url = _base_url(settings) or DEFAULT_GEMINI_URL
    api_key = _embedding_key(settings)
    if not api_key:
        raise ValueError("An API key is required for the Gemini embedding provider")
    model = settings.embedding_model
    qualified = model if model.startswith("models/") else f"models/{model}"
    task_type = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = await post_with_retry(
            client,
            f"{base_url}/{qualified}:batchEmbedContents",
            params={"key": api_key},
            json={"requests": [{
                "model": qualified,
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
                "outputDimensionality": GEMINI_DIMENSIONS,
            } for text in batch]},
        )
        ensure_success(response, "Gemini embedding")
        embeddings = response.json()["embeddings"]
        if len(embeddings) != len(batch):
            raise ValueError("Gemini returned an unexpected number of embeddings")
        # Truncated Gemini vectors are not unit length; normalise before storing.
        vectors.extend(_normalize(item["values"]) for item in embeddings)
    return vectors


async def create_embeddings(settings: AppSettings, texts: list[str],
                            kind: TextKind = "document") -> list[list[float]]:
    if not texts:
        return []
    provider = settings.embedding_provider
    if provider == "local":
        return [local_embed(text) for text in texts]
    if provider == "sentence-transformers":
        return await asyncio.to_thread(_embed_sentence_transformers, settings, texts, kind)

    timeout = httpx.Timeout(180, connect=30)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if provider == "ollama":
            return await _embed_ollama(settings, texts, client)
        if provider == "openai":
            return await _embed_openai(settings, texts, client)
        if provider == "gemini":
            return await _embed_gemini(settings, texts, kind, client)
    raise ValueError(f"Unknown embedding provider: {provider}")


async def create_query_embedding(settings: AppSettings, query: str) -> list[float]:
    """Embed the question.

    Feature hashing cannot bridge languages, so for that provider the query is
    blended with keyword-expanded variants to give the lexical signal something
    to match. A semantic model needs no such crutch, and blending only blurs the
    query, so it embeds the question as written.
    """
    if is_semantic(settings):
        vectors = await create_embeddings(settings, [query], kind="query")
        return vectors[0]

    variants = query_variants(query)
    vectors = await create_embeddings(settings, variants, kind="query")
    if len(vectors) == 1:
        return vectors[0]
    weights = [.55] + [(.45 / (len(vectors) - 1))] * (len(vectors) - 1)
    blended = [sum(weight * vector[index] for weight, vector in zip(weights, vectors))
               for index in range(len(vectors[0]))]
    return _normalize(blended)


def document_embedding_text(name: str, section: str | None, content: str) -> str:
    parts = [f"Document title: {name}"]
    if section:
        parts.append(f"Section: {section}")
    parts.append(f"Content: {content}")
    return "\n".join(parts)


async def reindex_all(settings: AppSettings) -> dict:
    chunks = database.rows(
        "SELECT c.id,c.content,c.section,d.name document_name FROM chunks c "
        "JOIN documents d ON d.id=c.document_id ORDER BY c.document_id,c.position"
    )
    if not chunks:
        probe = await create_embeddings(settings, ["embedding readiness check"])
        return {"chunks": 0, "dimensions": len(probe[0]) if probe else None}
    vectors = await create_embeddings(
        settings,
        [document_embedding_text(chunk["document_name"], chunk.get("section"), chunk["content"]) for chunk in chunks],
    )
    with database.connect() as db:
        db.executemany(
            "UPDATE chunks SET vector=? WHERE id=?",
            [(database.encode_vector(vector), chunk["id"]) for chunk, vector in zip(chunks, vectors)],
        )
    index.invalidate()
    return {"chunks": len(chunks), "dimensions": len(vectors[0]) if vectors else None}
