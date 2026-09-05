"""Who may call this API, and how often.

The application is built to be run by one person on their own machine, and the
defaults keep that true: with no `APP_AUTH_TOKEN` set, nothing is required and
nothing changes. The moment it is exposed - a laptop on a shared network, a
container with a published port - two things are needed and neither is worth
writing under pressure later.

Authentication is a single bearer token, compared in constant time. It is not
user management; it is the difference between "anyone who can reach the port
can read every document" and "not".

The rate limit is a fixed window per client, applied only to the two endpoints
that cost real money: answering a question calls a provider, and uploading a
document embeds every chunk of it. Everything else is a database read.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("adaptive_metric_rag.security")

# The page and its scripts are not data, and a browser cannot send a bearer
# header for its first request anyway: protecting them would only make the
# interface unreachable while every API route stayed exactly as protected.
PUBLIC_PATHS = ("/health", "/")
PUBLIC_PREFIXES = ("/assets/",)
RATE_LIMITED = {"/api/chat": "chat", "/api/chat/stream": "chat", "/api/documents": "upload"}
WINDOW_SECONDS = 60.0
# Clients that never come back would otherwise keep their window forever.
PRUNE_ABOVE = 10_000

_hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def auth_token() -> str:
    return os.getenv("APP_AUTH_TOKEN", "").strip()


def _limit(kind: str) -> int:
    """0 disables the limit for that kind."""
    default = "30" if kind == "chat" else "20"
    try:
        return max(0, int(os.getenv(f"APP_RATE_LIMIT_{kind.upper()}", default)))
    except ValueError:
        return int(default)


def client_id(request: Request) -> str:
    """Who to count against. The token when there is one, the address otherwise."""
    header = request.headers.get("authorization", "")
    if header:
        return f"token:{hash(header)}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def reset_limits() -> None:
    _hits.clear()


def over_limit(request: Request) -> tuple[bool, int, int]:
    """(rejected, limit, seconds until the window frees up) for this request."""
    kind = RATE_LIMITED.get(request.url.path)
    if kind is None or request.method == "GET":
        return False, 0, 0
    limit = _limit(kind)
    if not limit:
        return False, 0, 0
    now = time.monotonic()
    if len(_hits) > PRUNE_ABOVE:
        for key in [key for key, seen in _hits.items() if not seen or now - seen[-1] > WINDOW_SECONDS]:
            del _hits[key]
    window = _hits[(client_id(request), kind)]
    while window and now - window[0] > WINDOW_SECONDS:
        window.popleft()
    if len(window) >= limit:
        return True, limit, max(1, round(WINDOW_SECONDS - (now - window[0])))
    window.append(now)
    return False, limit, 0


def authorised(request: Request) -> bool:
    expected = auth_token()
    if not expected:
        return True
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES) or request.method == "OPTIONS":
        return True
    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented:
        return False
    return hmac.compare_digest(presented.strip(), expected)


async def guard(request: Request, call_next):
    """Reject what is not allowed before it reaches an endpoint."""
    if not authorised(request):
        logger.warning("rejected an unauthenticated request to %s", request.url.path)
        return JSONResponse({"detail": "Missing or invalid bearer token"}, status_code=401,
                            headers={"WWW-Authenticate": "Bearer"})
    rejected, limit, retry_after = over_limit(request)
    if rejected:
        logger.warning("rate limit reached for %s on %s", client_id(request), request.url.path)
        return JSONResponse(
            {"detail": f"Rate limit of {limit} requests per minute reached; retry in {retry_after}s"},
            status_code=429, headers={"Retry-After": str(retry_after)})
    return await call_next(request)
