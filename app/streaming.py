"""Streaming an answer as it is written.

Retrieval takes about a millisecond and generation takes seconds, so until the
model finished the reader had nothing at all - not even the sources, which were
known before the first token was requested. Streaming changes what the wait
looks like: the citations arrive as soon as retrieval is done, then the answer
arrives a phrase at a time.

The repair machinery stays where it was, after the stream. A truncated answer
cannot be detected until it stops, and rewriting text a reader has already seen
would be worse than the truncation. So a continuation is appended as further
tokens, and only a wrong-language answer - which is worth redoing - replaces
what was shown, through an explicit `replace` event.

Each provider streams a different shape: Ollama sends one JSON object per line,
OpenAI and Gemini send server-sent events carrying JSON. All three are read into
the same stream of text deltas here, so the endpoint and the browser know only
one format.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from .models import AppSettings
from .net import ProviderError, ensure_success, redact
from .providers import (_continuation_instruction, _finalize_answer, _looks_incomplete,
                        _repair_instruction, _wrong_language, build_prompt, local_answer)

logger = logging.getLogger("adaptive_metric_rag.streaming")

STREAM_TIMEOUT = httpx.Timeout(150.0, connect=15.0)
# Local answers are assembled instantly; they are still emitted in pieces so the
# browser has one rendering path rather than two.
LOCAL_CHUNK_CHARS = 60


def sse(event: str, data: Any) -> str:
    """One server-sent event. Newlines inside the payload would end it early."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _ollama_delta(line: str) -> tuple[str, str | None]:
    payload = json.loads(line)
    if payload.get("error"):
        raise ProviderError(f"Ollama generation failed: {payload['error']}")
    return payload.get("message", {}).get("content", ""), (payload.get("done_reason") or None
                                                           if payload.get("done") else None)


def _openai_delta(data: str) -> tuple[str, str | None]:
    payload = json.loads(data)
    choice = (payload.get("choices") or [{}])[0]
    return choice.get("delta", {}).get("content") or "", choice.get("finish_reason")


def _gemini_delta(data: str) -> tuple[str, str | None]:
    payload = json.loads(data)
    candidate = (payload.get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts), candidate.get("finishReason")


async def _sse_payloads(response: httpx.Response) -> AsyncIterator[str]:
    """The `data:` payloads of a server-sent event stream, one at a time."""
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield data


async def stream_answer(settings: AppSettings, question: str, chunks: list[dict[str, Any]],
                        language: str, history: list[dict[str, str]] | None = None,
                        structured: bool = False) -> AsyncIterator[tuple[str, str]]:
    """Yield ("delta", text) as the answer is written, then ("done", full answer).

    A wrong-language answer yields ("replace", corrected) before it finishes; a
    truncated one yields further deltas. Both happen after the stream, because
    neither can be known before it ends.
    """
    if settings.provider == "local":
        answer = local_answer(question, chunks, language)
        for start in range(0, len(answer), LOCAL_CHUNK_CHARS):
            yield "delta", answer[start:start + LOCAL_CHUNK_CHARS]
        yield "done", answer
        return

    system, user = build_prompt(question, chunks, settings.system_prompt, language, history,
                                structured=structured)
    collected: list[str] = []
    finish_reason = ""

    async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
        async for piece, reason in _provider_stream(settings, client, system, user):
            if piece:
                collected.append(piece)
                yield "delta", piece
            if reason:
                finish_reason = reason

        answer = "".join(collected)
        if not answer.strip():
            raise ProviderError("the provider streamed an empty answer")

        # Structured output is JSON, so it was never readable as it streamed;
        # the caller replaces what was shown with the parsed answer text.
        if _wrong_language(language, answer) and not structured:
            repaired = await _repair(settings, client, system, user, answer, language)
            if repaired:
                answer = repaired
                yield "replace", answer
        elif not structured:
            for _ in range(2):
                if not _looks_incomplete(question, answer, finish_reason,
                                         settings.strict_multipart_answers):
                    break
                piece = await _continue(settings, client, system, user, answer, language)
                if not piece:
                    break
                answer = answer.rstrip() + " " + piece.lstrip()
                finish_reason = ""
                yield "delta", " " + piece.lstrip()
    yield "done", answer if structured else _finalize_answer(answer)


async def _provider_stream(settings: AppSettings, client: httpx.AsyncClient, system: str,
                           user: str) -> AsyncIterator[tuple[str, str | None]]:
    provider = settings.provider
    if provider == "ollama":
        base = (settings.base_url or "http://host.docker.internal:11434").rstrip("/")
        body = {"model": settings.model, "stream": True, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": settings.temperature, "num_predict": settings.max_tokens}}
        async with client.stream("POST", f"{base}/api/chat", json=body) as response:
            await _ensure_stream_success(response, "Ollama generation")
            async for line in response.aiter_lines():
                if line.strip():
                    yield _ollama_delta(line)
        return
    if provider == "openai":
        base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
        body = {"model": settings.model, "temperature": settings.temperature,
                "max_tokens": settings.max_tokens, "stream": True,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {settings.api_key}"}
        async with client.stream("POST", f"{base}/chat/completions", json=body, headers=headers) as response:
            await _ensure_stream_success(response, "OpenAI generation")
            async for data in _sse_payloads(response):
                yield _openai_delta(data)
        return
    if provider == "gemini":
        base = (settings.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        body = {"system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": settings.temperature,
                                     "maxOutputTokens": settings.max_tokens}}
        url = f"{base}/models/{settings.model}:streamGenerateContent"
        params = {"key": settings.api_key, "alt": "sse"}
        async with client.stream("POST", url, json=body, params=params) as response:
            await _ensure_stream_success(response, "Gemini generation")
            async for data in _sse_payloads(response):
                yield _gemini_delta(data)
        return
    raise ValueError(f"Provider {settings.provider} cannot stream")


async def _ensure_stream_success(response: httpx.Response, what: str) -> None:
    """Read the body before reporting a failed stream; it holds the reason."""
    if response.is_success:
        return
    await response.aread()
    ensure_success(response, what)


async def _repair(settings: AppSettings, client: httpx.AsyncClient, system: str, user: str,
                  answer: str, language: str) -> str:
    from .providers import complete

    try:
        return await complete(settings, system,
                              f"{user}\n\nPrevious answer:\n{answer}\n\n{_repair_instruction(language)}",
                              temperature=min(settings.temperature, .2), max_tokens=settings.max_tokens)
    except Exception as exc:
        logger.warning("streamed answer could not be repaired: %s", redact(exc))
        return ""


async def _continue(settings: AppSettings, client: httpx.AsyncClient, system: str, user: str,
                    answer: str, language: str) -> str:
    from .providers import complete

    try:
        return await complete(settings, system,
                              f"{user}\n\nAnswer so far:\n{answer}\n\n{_continuation_instruction(language)}",
                              temperature=min(settings.temperature, .2),
                              max_tokens=min(settings.max_tokens, 512))
    except Exception as exc:
        logger.warning("streamed answer could not be continued: %s", redact(exc))
        return ""
