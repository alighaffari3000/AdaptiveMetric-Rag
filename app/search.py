"""The retrieval pipeline, from a question to the chunks worth answering from.

Keeping the stages in one place means the API, the evaluation harness, and any
future caller all measure and serve the same pipeline.

    question
      -> embed
      -> first stage: dense + BM25 candidates, fused by rank
      -> second stage: rerank the shortlist (optional)
      -> evidence gate: answer from these chunks, or abstain
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .embeddings import create_query_embedding, is_semantic
from .models import AppSettings, QueryAnalysis
from .rerank import RerankUnavailable, blend, rerank_scores
from .retrieval import RetrievalResult, retrieve, score_confidence, select_grounded

logger = logging.getLogger("adaptive_metric_rag.search")


@dataclass
class SearchResult:
    chunks: list[dict[str, Any]]
    analysis: QueryAnalysis
    confidence: float
    early_exit: bool
    evidence_found: bool
    reranked: bool
    timings_ms: dict[str, int] = field(default_factory=dict)


async def search(settings: AppSettings, query: str, filters: dict[str, Any] | None = None) -> SearchResult:
    timings: dict[str, int] = {}

    started = time.perf_counter()
    query_vector = await create_query_embedding(settings, query)
    timings["embed"] = round((time.perf_counter() - started) * 1000)

    # Retrieve deeper than we will show whenever a reranker can reorder the list.
    reranking_possible = settings.rerank_enabled
    depth = max(settings.context_count, settings.rerank_top_n) if reranking_possible else settings.context_count
    # Rank fusion is the right call only when the dense signal carries meaning.
    # Feature hashing produces a dense score that ranks poorly, and fusing by
    # rank would hand it its full intent weight anyway.
    semantic = is_semantic(settings)
    dense_weight = settings.semantic_dense_weight if semantic else None
    fusion = "rank" if semantic else "linear"

    started = time.perf_counter()
    result: RetrievalResult = await asyncio.to_thread(
        retrieve, query, settings.candidate_count, depth, filters, query_vector, dense_weight, fusion
    )
    timings["retrieve"] = round((time.perf_counter() - started) * 1000)

    chunks = result.chunks
    confidence = result.confidence
    reranked = False

    should_rerank = (
        reranking_possible
        and len(chunks) > 1
        and not (result.early_exit and settings.enable_early_exit)
    )
    if should_rerank:
        shortlist = chunks[:settings.rerank_top_n]
        started = time.perf_counter()
        try:
            scores = await rerank_scores(settings, query, shortlist)
            chunks = blend(shortlist, scores, settings.rerank_weight) + chunks[settings.rerank_top_n:]
            reranked = True
        except RerankUnavailable as exc:
            logger.warning("reranking skipped, keeping the fusion ordering: %s", exc)
        except Exception as exc:  # a reranker must never take down the answer
            logger.warning("reranking failed, keeping the fusion ordering: %r", exc, exc_info=True)
        timings["rerank"] = round((time.perf_counter() - started) * 1000)

    chunks = chunks[:settings.context_count]
    if reranked:
        confidence = score_confidence(chunks, result.analysis.query_tokens or result.analysis.keywords,
                                      result.standout)

    gated = RetrievalResult(chunks, result.analysis, confidence, result.early_exit, result.standout)
    evidence_found, grounded = select_grounded(gated)
    return SearchResult(
        chunks=grounded,
        analysis=result.analysis,
        confidence=round(confidence, 3),
        early_exit=result.early_exit,
        evidence_found=evidence_found,
        reranked=reranked,
        timings_ms=timings,
    )
