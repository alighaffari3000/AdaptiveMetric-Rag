from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import database
from .documents import (ALLOWED, content_digest, find_duplicate, ingest, parent_windows,
                        refresh_chunk_tokens)
from .embeddings import embedding_info, is_semantic, reindex_all
from .index import index as chunk_index
from .net import ProviderError, redact
from .models import AppSettings, ChatRequest, ChatResponse, Citation, SettingsView
from .providers import generate, local_answer
from .retrieval import best_evidence
from .search import search
from .text import NORMALIZER_VERSION


logger = logging.getLogger("adaptive_metric_rag")


CITATION_REFRESH_KEY = "citation_highlights_refreshed_at"
NORMALIZER_KEY = "text_normalizer_version"


def refresh_saved_citation_highlights() -> None:
    """Upgrade saved conversations to the current answer-aware evidence selector.

    This is a one-shot migration, not startup work. It used to rewrite the whole
    message history on every boot, which grew slower with every conversation.
    """
    messages = database.rows(
        "SELECT id,conversation_id,role,content,citations FROM messages ORDER BY conversation_id,created_at"
    )
    last_question: dict[str, str] = {}
    for message in messages:
        if message["role"] == "user":
            last_question[message["conversation_id"]] = message["content"]
            continue
        question = last_question.get(message["conversation_id"], "")
        try:
            citations = json.loads(message["citations"] or "[]")
        except json.JSONDecodeError:
            continue
        changed = False
        for citation in citations:
            chunk = database.row("SELECT content FROM chunks WHERE id=?", (citation.get("chunk_id", ""),))
            if chunk:
                claim = claim_for_citation(message["content"], int(citation.get("id", 0)))
                citation["highlight"] = best_evidence(question, chunk["content"], claim)
                changed = True
        if changed:
            database.execute("UPDATE messages SET citations=? WHERE id=?", (database.json_value(citations), message["id"]))


async def upgrade_to_current_normalizer(settings: AppSettings) -> None:
    """Re-derive what the text normalizer decides, once, after it changes.

    Chunk tokens are computed at ingestion, so a library indexed before the
    normalizer existed holds `۱۴۰۳` where a question now produces `1403`: BM25
    would silently stop matching them. Feature-hashed vectors are built from the
    same tokens and go stale with them, so they are rebuilt too - offline and
    free. Vectors from a real embedding model do not depend on tokenization and
    are left untouched, so no start-up ever calls a paid API.
    """
    changed = await asyncio.to_thread(refresh_chunk_tokens)
    if changed and not is_semantic(settings):
        await reindex_all(settings)
    if changed:
        logger.info("re-derived tokens for %d chunks after a normalizer change", changed)


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.init_db()
    if not database.meta_get(CITATION_REFRESH_KEY):
        await asyncio.to_thread(refresh_saved_citation_highlights)
        database.meta_set(CITATION_REFRESH_KEY, datetime.now(timezone.utc).isoformat())
    if database.meta_get(NORMALIZER_KEY) != NORMALIZER_VERSION:
        await upgrade_to_current_normalizer(load_settings())
        database.meta_set(NORMALIZER_KEY, NORMALIZER_VERSION)
    await asyncio.to_thread(chunk_index.reload)
    yield


app = FastAPI(title="AdaptiveMetric RAG", version="1.0.0", lifespan=lifespan)
static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/assets", StaticFiles(directory=static_dir), name="assets")


def no_evidence_answer(language: str) -> str:
    if language == "fa":
        return "پاسخ این سؤال در منابع موجود پیدا نشد. لطفاً سند مرتبط‌تری اضافه کنید یا سؤال را دقیق‌تر بنویسید."
    return "The answer to this question was not found in the available sources. Please add a relevant document or make the question more specific."


def claim_for_citation(answer: str, citation_id: int) -> str:
    marker = f"[{citation_id}]"
    sentences = re.split(r"(?<=[.!?؟؛])\s+|\n+", answer)
    matching = [sentence for sentence in sentences if marker in sentence]
    return " ".join(matching) if matching else answer


def answer_abstained(answer: str) -> bool:
    normalized = answer.strip().lower()
    phrases = (
        "پاسخ این سؤال در منابع موجود پیدا نشد",
        "پاسخ این سوال در منابع موجود پیدا نشد",
        "اطلاعات کافی برای پاسخ در منابع وجود ندارد",
        "the answer was not found in the available sources",
        "the provided sources do not contain enough information to answer",
    )
    return any(phrase in normalized for phrase in phrases)


def load_settings() -> AppSettings:
    saved = database.row("SELECT value FROM settings WHERE id=1")
    base = json.loads(saved["value"]) if saved else {}
    if not base.get("api_key"):
        provider = base.get("provider", "local")
        if provider == "openai":
            base["api_key"] = os.getenv("OPENAI_API_KEY", "")
        elif provider == "gemini":
            base["api_key"] = os.getenv("GEMINI_API_KEY", "")
    if not base.get("embedding_api_key"):
        embedding_provider = base.get("embedding_provider", "local")
        if embedding_provider == "openai":
            base["embedding_api_key"] = os.getenv("OPENAI_API_KEY", "")
        elif embedding_provider == "gemini":
            base["embedding_api_key"] = os.getenv("GEMINI_API_KEY", "")
    if base.get("provider") == "ollama" and not base.get("base_url"):
        base["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    if base.get("embedding_provider") == "ollama" and not base.get("embedding_base_url"):
        base["embedding_base_url"] = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    return AppSettings(**base)


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health():
    docs = database.row("SELECT COUNT(*) count FROM documents")
    chunks = database.row("SELECT COUNT(*) count FROM chunks")
    return {"status": "ok", "version": app.version, "documents": docs["count"], "chunks": chunks["count"]}


@app.get("/api/system/info")
async def system_info():
    info = embedding_info(load_settings())
    # A cold index reloads from the database here; keep that off the event loop.
    snapshot = await asyncio.to_thread(chunk_index.snapshot)
    if snapshot.size:
        info["dimensions"] = int(snapshot.matrix.shape[1])
    return {"embedding": info, "index": {"chunks": snapshot.size, "stale_vectors": snapshot.stale_vectors}}


@app.get("/api/settings", response_model=SettingsView)
async def get_settings():
    current = load_settings()
    payload = current.model_dump()
    payload["has_api_key"] = bool(current.api_key)
    payload["has_embedding_api_key"] = bool(current.embedding_api_key)
    payload["api_key"] = ""
    payload["embedding_api_key"] = ""
    return SettingsView(**payload)


@app.put("/api/settings", response_model=SettingsView)
async def save_settings(settings: AppSettings):
    existing = load_settings()
    if not settings.api_key:
        settings.api_key = existing.api_key
    if not settings.embedding_api_key:
        settings.embedding_api_key = existing.embedding_api_key
    embedding_changed = (
        settings.embedding_provider != existing.embedding_provider
        or settings.embedding_model != existing.embedding_model
        or settings.embedding_base_url != existing.embedding_base_url
    )
    if embedding_changed:
        try:
            await reindex_all(settings)
        except (httpx.HTTPError, ProviderError, KeyError, ValueError) as exc:
            raise HTTPException(502, f"Embedding reindex failed; settings were not changed: {redact(exc)}") from exc
    database.execute("INSERT INTO settings(id,value) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET value=excluded.value", (database.json_value(settings.model_dump()),))
    payload = settings.model_dump()
    payload.update(api_key="", has_api_key=bool(settings.api_key),
                   embedding_api_key="", has_embedding_api_key=bool(settings.embedding_api_key))
    return SettingsView(**payload)


@app.post("/api/settings/test")
async def test_provider():
    settings = load_settings()
    if settings.provider == "local":
        return {"ok": True, "message": "Local extractive provider is ready."}
    try:
        if settings.provider == "ollama":
            async with httpx.AsyncClient(timeout=30) as client:
                base_url = settings.base_url.rstrip('/')
                tags_response = await client.get(f"{base_url}/api/tags")
                tags_response.raise_for_status()
                installed = [item.get("name") or item.get("model") for item in tags_response.json().get("models", [])]
                if settings.model not in installed:
                    raise ValueError(f"Model '{settings.model}' is not installed in Ollama")
                response = await client.post(f"{base_url}/api/chat", json={
                    "model": settings.model, "stream": False, "think": False,
                    "messages": [{"role": "user", "content": "Reply with OK only."}],
                    "options": {"temperature": 0, "num_predict": 8},
                })
                response.raise_for_status()
        elif not settings.api_key:
            raise ValueError("API key is missing")
        return {"ok": True, "message": f"{settings.provider.title()} configuration is ready."}
    except Exception as exc:
        raise HTTPException(400, f"Connection failed: {redact(exc)}") from exc


@app.get("/api/embeddings/ollama-models")
async def ollama_embedding_models(base_url: str = ""):
    settings = load_settings()
    base_url = (base_url or settings.embedding_base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
        models = [item.get("name") or item.get("model") for item in response.json().get("models", [])]
        models = [model for model in models if model]
        embedding_hints = ("embed", "bge", "nomic", "mxbai", "e5", "gte", "snowflake", "jina")
        likely_embedding_models = [model for model in models if any(hint in model.lower() for hint in embedding_hints)]
        return {"models": likely_embedding_models or models, "all_models_count": len(models), "base_url": base_url}
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(502, f"Could not read Ollama models: {exc}") from exc


@app.get("/api/ollama/generation-models")
async def ollama_generation_models(base_url: str = ""):
    settings = load_settings()
    base_url = (base_url or settings.base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
        models = [item.get("name") or item.get("model") for item in response.json().get("models", [])]
        embedding_hints = ("embed", "bge", "nomic", "mxbai", "e5", "gte", "snowflake", "jina")
        generation_models = [model for model in models if model and not any(hint in model.lower() for hint in embedding_hints)]
        return {"models": generation_models, "base_url": base_url}
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(502, f"Could not read Ollama models: {exc}") from exc


@app.get("/api/documents")
async def list_documents():
    return database.rows("SELECT * FROM documents ORDER BY created_at DESC")


async def run_ingestion(document_id: str, filename: str, content_type: str, payload: bytes,
                        settings: AppSettings) -> None:
    """Ingest one document, recording where it got to and why it stopped."""

    def progress(fraction: float) -> None:
        database.execute("UPDATE documents SET progress=? WHERE id=?", (fraction, document_id))

    try:
        await ingest(filename, content_type, payload, settings, document_id=document_id, progress=progress)
    except Exception as exc:
        logger.error("ingestion failed for %s: %r", filename, exc, exc_info=True)
        database.execute("UPDATE documents SET status='failed', error=? WHERE id=?",
                         (redact(exc)[:500], document_id))


@app.post("/api/documents", status_code=202)
async def upload_document(background: BackgroundTasks, file: UploadFile = File(...)):
    """Accept the file, then parse it outside the request.

    A large PDF used to hold the HTTP connection open for its whole extraction
    and embedding run, with nothing to show for it until the end. The document
    row now appears immediately with `status="processing"`, and
    `/api/documents/{id}/status` reports how far it has got.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported format. Use: {', '.join(sorted(ALLOWED))}")
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "The uploaded file is empty")
    if len(payload) > 30 * 1024 * 1024:
        raise HTTPException(413, "Maximum file size is 30 MB")
    duplicate = find_duplicate(content_digest(payload))
    if duplicate:
        raise HTTPException(
            409,
            f"This file is already in the library as '{duplicate['name']}'. "
            "Delete it first to replace it.",
        )
    settings = load_settings()
    filename = file.filename or "document"
    document_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    database.execute(
        "INSERT INTO documents(id,name,type,size,chunks,created_at,metadata,status,progress,error) "
        "VALUES(?,?,?,?,0,?,?,'processing',0.0,'')",
        (document_id, filename, file.content_type or "application/octet-stream", len(payload), now,
         database.json_value({"sha256": content_digest(payload)})),
    )
    background.add_task(run_ingestion, document_id, filename, file.content_type or "", payload, settings)
    return {"id": document_id, "name": filename, "type": file.content_type, "size": len(payload),
            "chunks": 0, "created_at": now, "status": "processing", "progress": 0.0}


@app.get("/api/documents/{document_id}/status")
async def document_status(document_id: str):
    document = database.row(
        "SELECT id,name,status,progress,error,chunks,metadata FROM documents WHERE id=?", (document_id,))
    if not document:
        raise HTTPException(404, "Document not found")
    try:
        metadata = json.loads(document.pop("metadata") or "{}")
    except json.JSONDecodeError:
        metadata = {}
    document["warnings"] = metadata.get("warnings", [])
    return document


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    found = database.row("SELECT id FROM documents WHERE id=?", (document_id,))
    if not found:
        raise HTTPException(404, "Document not found")
    database.execute("DELETE FROM documents WHERE id=?", (document_id,))
    chunk_index.invalidate()
    return {"ok": True}


@app.get("/api/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    chunk = database.row(
        "SELECT c.id,c.document_id,c.page,c.page_end,c.section,c.section_path,c.content,d.name document_name "
        "FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.id=?",
        (chunk_id,),
    )
    if not chunk:
        raise HTTPException(404, "Chunk not found")
    return chunk


def conversation_history(conversation_id: str, turns: int) -> list[dict[str, str]]:
    """The last few turns of an ongoing conversation, oldest first.

    Read straight from storage rather than from the client, so the rewriter and
    the answer prompt see what was actually said, and a request cannot inject a
    history that never happened.
    """
    if not conversation_id or turns <= 0:
        return []
    recent = database.rows(
        "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (conversation_id, turns),
    )
    return [{"role": message["role"], "content": message["content"]} for message in reversed(recent)]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    started = time.perf_counter()
    settings = load_settings()
    history = conversation_history(request.conversation_id or "", settings.history_turns)
    try:
        result = await search(settings, request.message, request.filters, history,
                              request.conversation_id or "")
    except (httpx.HTTPError, ProviderError, KeyError, ValueError) as exc:
        raise HTTPException(502, f"Retrieval failed: {redact(exc)}") from exc

    evidence_found = result.evidence_found
    grounded_chunks = result.chunks
    # Retrieval scored small chunks; the model answers from the windows around
    # them, so a fact never arrives without the sentences that qualify it.
    context = await asyncio.to_thread(parent_windows, grounded_chunks) if grounded_chunks else []
    if evidence_found:
        try:
            answer = await generate(settings, request.message, context, result.analysis.language, history)
        except (httpx.HTTPError, ProviderError, KeyError, ValueError) as first_exc:
            logger.warning("Generation failed; retrying with reduced context: %r", first_exc, exc_info=True)
            try:
                answer = await generate(settings, request.message, context[:2],
                                        result.analysis.language, history)
            except (httpx.HTTPError, ProviderError, KeyError, ValueError) as retry_exc:
                logger.error("Generation retry failed; using grounded extractive fallback: %r", retry_exc, exc_info=True)
                answer = local_answer(request.message, context, result.analysis.language)
    else:
        answer = no_evidence_answer(result.analysis.language)
    if evidence_found and answer_abstained(answer):
        evidence_found = False
        grounded_chunks = []
    # Highlighting reads the retrieved question: for a follow-up, the words that
    # located the passage are the ones the conversation supplied, not «و مبلغش؟».
    highlight_query = result.plan.question if result.plan else request.message
    citations = [Citation(
        id=i, document_id=chunk["document_id"], document_name=chunk["document_name"], chunk_id=chunk["id"],
        page=chunk.get("page"), page_end=chunk.get("page_end"), section=chunk.get("section"),
        section_path=chunk.get("section_path") or "", excerpt=chunk["content"][:420],
        highlight=best_evidence(highlight_query, chunk["content"], claim_for_citation(answer, i)), score=chunk["score"],
    ) for i, chunk in enumerate(grounded_chunks, 1)]
    conversation_id = request.conversation_id or uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    if not request.conversation_id:
        database.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)", (conversation_id, request.message[:72], now, now))
    database.execute("INSERT INTO messages(id,conversation_id,role,content,citations,created_at) VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex, conversation_id, "user", request.message, "[]", now))
    database.execute("INSERT INTO messages(id,conversation_id,role,content,citations,created_at) VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex, conversation_id, "assistant", answer, database.json_value([c.model_dump() for c in citations]), now))
    database.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
    return ChatResponse(
        conversation_id=conversation_id, answer=answer, citations=citations, analysis=result.analysis,
        confidence=result.confidence, latency_ms=round((time.perf_counter() - started) * 1000),
        provider=settings.provider,
        early_exit=(result.early_exit and settings.enable_early_exit
                    and result.confidence >= settings.confidence_threshold),
        evidence_found=evidence_found, reranked=result.reranked, timings_ms=result.timings_ms,
    )


@app.get("/api/conversations")
async def conversations():
    return database.rows("SELECT * FROM conversations ORDER BY updated_at DESC LIMIT 50")


@app.get("/api/conversations/{conversation_id}")
async def conversation(conversation_id: str):
    return database.rows("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (conversation_id,))


@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str):
    current = database.row("SELECT * FROM conversations WHERE id=?", (conversation_id,))
    if not current:
        raise HTTPException(404, "Conversation not found")
    messages = database.rows(
        "SELECT id,role,content,citations,created_at FROM messages WHERE conversation_id=? ORDER BY created_at",
        (conversation_id,),
    )
    for message in messages:
        try:
            message["citations"] = json.loads(message["citations"] or "[]")
        except json.JSONDecodeError:
            message["citations"] = []
    settings = load_settings()
    payload = {
        "format": "adaptivemetric-rag-conversation",
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "conversation": current,
        "runtime": {
            "generation_provider": settings.provider,
            "generation_model": settings.model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
        },
        "messages": messages,
    }
    filename = f"adaptive-rag-{conversation_id[:10]}.json"
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    found = database.row("SELECT id FROM conversations WHERE id=?", (conversation_id,))
    if not found:
        raise HTTPException(404, "Conversation not found")
    database.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    return {"ok": True, "deleted": 1}


@app.delete("/api/conversations")
async def delete_all_conversations():
    count = database.row("SELECT COUNT(*) count FROM conversations")["count"]
    database.execute("DELETE FROM conversations")
    return {"ok": True, "deleted": count}
