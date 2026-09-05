"""Second-stage reranking.

First-stage retrieval reads a question and a chunk separately and compares two
vectors, so it can only ever measure topical closeness. A reranker reads the
question and the chunk together and judges whether that chunk actually answers
this question, which is a different and much harder question to get right.

Two backends, because the right one depends on what the deployment already has:

- `cross-encoder` runs a local scoring model and needs no network
- `llm` asks the configured generation provider to score the shortlist in one
  call, which needs no extra model but does spend a request per query
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from typing import Any

from .models import AppSettings

logger = logging.getLogger("adaptive_metric_rag.rerank")

DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"
PASSAGE_CHARS = 700

_cross_encoders: dict[str, Any] = {}
_cross_encoder_lock = threading.Lock()


class RerankUnavailable(RuntimeError):
    """The reranker could not run; the caller should keep the fusion ordering."""


def _load_cross_encoder(model_name: str):
    with _cross_encoder_lock:
        model = _cross_encoders.get(model_name)
        if model is not None:
            return model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankUnavailable(
                "The cross-encoder reranker needs sentence-transformers: "
                "pip install --extra-index-url https://download.pytorch.org/whl/cpu "
                "-r requirements-local-embeddings.txt. The 'llm' backend needs no local model."
            ) from exc
        logger.info("loading cross-encoder %s (first use downloads it)", model_name)
        model = CrossEncoder(model_name)
        _cross_encoders[model_name] = model
        return model


def _score_cross_encoder(model_name: str, query: str, passages: list[str]) -> list[float]:
    model = _load_cross_encoder(model_name)
    pairs = [(query, passage) for passage in passages]
    scores = model.predict(pairs, show_progress_bar=False)
    return [float(score) for score in scores]


def build_llm_prompt(query: str, passages: list[str]) -> tuple[str, str]:
    numbered = "\n\n".join(
        f"[{position}] {passage[:PASSAGE_CHARS]}" for position, passage in enumerate(passages, 1)
    )
    system = (
        "You rate how well each passage answers a question. Judge only whether the passage "
        "contains the information the question asks for, not how well written it is. A passage "
        "about the right topic that does not contain the answer scores low. The question and the "
        "passages may be in different languages; that is never a reason to score low."
    )
    user = (
        f"Question:\n{query}\n\nPassages:\n{numbered}\n\n"
        f"Score every passage from 0 to 10. Reply with JSON only, in the form "
        f'{{"scores": [{{"id": 1, "score": 7}}, ...]}}, covering all {len(passages)} passages.'
    )
    return system, user


def parse_llm_scores(raw: str, count: int) -> list[float]:
    """Read the model's JSON, tolerating code fences and surrounding prose."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RerankUnavailable("the reranking model did not return JSON")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RerankUnavailable(f"the reranking model returned invalid JSON: {exc}") from exc
    entries = payload.get("scores")
    if not isinstance(entries, list) or not entries:
        raise RerankUnavailable("the reranking model returned no scores")

    scores = [0.0] * count
    seen = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            position = int(entry.get("id", 0)) - 1
            value = float(entry.get("score", 0))
        except (TypeError, ValueError):
            continue
        if 0 <= position < count:
            scores[position] = max(0.0, min(10.0, value)) / 10.0
            seen += 1
    if not seen:
        raise RerankUnavailable("the reranking model scored none of the passages")
    return scores


async def _score_llm(settings: AppSettings, query: str, passages: list[str]) -> list[float]:
    from .providers import complete

    system, user = build_llm_prompt(query, passages)
    model = settings.rerank_model or settings.model
    raw = await complete(settings, system, user, model=model, temperature=0.0, max_tokens=1200)
    return parse_llm_scores(raw, len(passages))


async def rerank_scores(settings: AppSettings, query: str, chunks: list[dict[str, Any]]) -> list[float]:
    """Return a relevance score in [0, 1] for each chunk, in the given order."""
    if not chunks:
        return []
    passages = [chunk["content"] for chunk in chunks]
    if settings.rerank_backend == "cross-encoder":
        model_name = settings.rerank_model or DEFAULT_CROSS_ENCODER
        raw = await asyncio.to_thread(_score_cross_encoder, model_name, query, passages)
        lowest, highest = min(raw), max(raw)
        if highest - lowest < 1e-9:
            return [0.5] * len(raw)
        return [(score - lowest) / (highest - lowest) for score in raw]
    if settings.rerank_backend == "llm":
        if settings.provider == "local":
            raise RerankUnavailable(
                "The llm reranker needs a generation provider; the local extractive mode cannot score passages."
            )
        return await _score_llm(settings, query, passages)
    raise RerankUnavailable(f"Unknown rerank backend: {settings.rerank_backend}")


def blend(chunks: list[dict[str, Any]], rerank: list[float], weight: float) -> list[dict[str, Any]]:
    """Combine the reranker's judgement with the fusion score.

    Keeping a share of the fusion score matters: the reranker sees a truncated
    passage and nothing about exact identifiers, dates or document metadata, so
    the signals the adaptive metric measured stay in the decision.
    """
    ordered = []
    for chunk, score in zip(chunks, rerank):
        updated = dict(chunk)
        updated["rerank_score"] = round(float(score), 4)
        updated["fusion_score"] = chunk["score"]
        updated["score"] = round(weight * float(score) + (1 - weight) * chunk["score"], 6)
        ordered.append(updated)
    ordered.sort(key=lambda item: item["score"], reverse=True)
    return ordered
