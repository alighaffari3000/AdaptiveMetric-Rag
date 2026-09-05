"""Logs that can be searched, and a request id that ties them together.

One question touches retrieval, embedding, reranking, generation and
verification, each logging under its own name. When something goes wrong in
production the useful question is "what happened during that request", and the
previous format - unstructured lines, no correlation - could not answer it.

Every record is one JSON object carrying the request id of the call it belongs
to. The id comes from the caller's `X-Request-ID` when it sends one, so a proxy
or a client can follow a request across services, and is generated otherwise.
The response carries it back.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid

from fastapi import Request

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info",
    "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per record, with whatever the caller attached to it."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) +
                    f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        current = request_id.get()
        if current:
            payload["request_id"] = current
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Install the format and level this deployment asked for.

    `APP_LOG_FORMAT=text` keeps the readable format for a terminal; the default
    is JSON, because the default deployment is a container. The level defaults
    to info: warning was hiding the ordinary course of events, which is exactly
    what is wanted when something goes wrong an hour later.
    """
    level = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    if os.getenv("APP_LOG_FORMAT", "json").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    # Access logs would double every line the middleware already records.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


async def tag_requests(request: Request, call_next):
    """Give every request an id, log how it went, and hand the id back."""
    incoming = request.headers.get("x-request-id", "").strip()
    token = request_id.set(incoming[:64] or uuid.uuid4().hex[:16])
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id.get()
        return response
    finally:
        duration = round((time.perf_counter() - started) * 1000, 1)
        logging.getLogger("adaptive_metric_rag.request").info(
            "%s %s -> %s", request.method, request.url.path, status,
            extra={"method": request.method, "path": request.url.path,
                   "status": status, "duration_ms": duration},
        )
        request_id.reset(token)
