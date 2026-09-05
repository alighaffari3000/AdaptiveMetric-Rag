"""Shared HTTP retry policy for outbound provider calls.

Embedding a document and reranking a shortlist are both sequences of requests
where one dropped connection or one rate-limit response should not discard the
work already done. Rate limits in particular are routine on hosted providers and
are answered by waiting, not by failing.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re

import httpx

logger = logging.getLogger("adaptive_metric_rag.net")

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY = 0.75
RETRY_MAX_DELAY = 30.0
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

_SECRET_PATTERNS = (
    # key=..., api_key=..., token=... in a URL query string or anywhere in prose;
    # the lookbehind keeps "monkey=banana" out of it.
    re.compile(r"((?:^|(?<=[?&\s'\"(,;]))(?:key|api_key|apikey|access_token|token)=)[^&\s'\")]+", re.IGNORECASE),
    # Authorization: Bearer <token>
    re.compile(r"(bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE),
)


def redact(text: object) -> str:
    """Strip credentials from text that is about to be logged or returned.

    Gemini passes the API key as a query parameter, so httpx puts it into the
    URL of every error it raises, and those errors reach logs and API responses.
    Bodies and headers are covered too, in case a provider echoes them.
    """
    cleaned = str(text)
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(r"\1***", cleaned)
    return cleaned


class ProviderError(RuntimeError):
    """A provider call failed. The message never carries an API key."""

    def __init__(self, message: str):
        super().__init__(redact(message))


def ensure_success(response: httpx.Response, what: str) -> None:
    """Replace `response.raise_for_status()` on provider calls.

    httpx puts the request URL into its error text, and for Gemini that URL
    carries the API key as a query parameter.
    """
    if response.is_success:
        return
    detail = response.text[:300] if response.text else ""
    raise ProviderError(f"{what} failed with HTTP {response.status_code}: {detail}")


def _retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return min(RETRY_MAX_DELAY, float(header))
    except ValueError:
        return None


async def post_with_retry(client: httpx.AsyncClient, url: str, *, what: str = "request",
                          attempts: int = RETRY_ATTEMPTS, **kwargs) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(attempts):
        response = None
        try:
            response = await client.post(url, **kwargs)
            if response.status_code not in RETRYABLE_STATUS or attempt == attempts - 1:
                return response
            last = httpx.HTTPStatusError(f"HTTP {response.status_code}",
                                         request=response.request, response=response)
        except httpx.TransportError as exc:
            last = exc
            if attempt == attempts - 1:
                break
        delay = _retry_after(response)
        if delay is None:
            delay = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt))
            delay += random.uniform(0, 0.4) if delay else 0
        logger.warning("%s failed (%s); retrying in %.1fs", what, redact(last), delay)
        await asyncio.sleep(delay)
    assert last is not None
    raise last
