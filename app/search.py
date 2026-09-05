"""The retrieval pipeline, from a question to the chunks worth answering from.

Keeping the stages in one place means the API, the evaluation harness, and any
future caller all measure and serve the same pipeline.

    question
      -> rewrite: make it standalone, add alternative phrasings
      -> embed
      -> first stage: dense + BM25 candidates, fused by rank
      -> multi-query: merge the phrasings by rank
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
from .rewrite import QueryPlan, plan_query
from .retrieval import RetrievalResult, fuse_results, retrieve, score_confidence, select_grounded
from .text import language_of

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
    plan: QueryPlan | None = None


async def search(settings: AppSettings, query: str, filters: dict[str, Any] | None = None,
                 history: list[dict[str, str]] | None = None, conversation_id: str = "") -> SearchResult:
    timings: dict[str, int] = {}

    started = time.perf_counter()
    plan = await plan_query(settings, query, history, conversation_id)
    timings["rewrite"] = round((time.perf_counter() - started) * 1000)
    questions = list(dict.fromkeys([plan.question] + plan.variants))

    started = time.perf_counter()
    query_vectors = await asyncio.gather(
        *(create_query_embedding(settings, question) for question in questions)
    )
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
    results: list[RetrievalResult] = await asyncio.gather(
        *(asyncio.to_thread(retrieve, question, settings.candidate_count, depth, filters,
                            vector, dense_weight, fusion)
          for question, vector in zip(questions, query_vectors))
    )
    result = fuse_results(results, depth)
    timings["retrieve"] = round((time.perf_counter() - started) * 1000)

    if plan.rewritten:
        result.analysis.rewritten_from = plan.original
        result.analysis.rewrite_source = plan.source
        # The rewrite exists to be retrieved with; the answer is still written
        # in the language the person used. Carrying Persian topic words into an
        # English follow-up must not switch the reply to Persian.
        result.analysis.language = language_of(plan.original)

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
            scores = await rerank_scores(settings, plan.question, shortlist)
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

    gated = RetrievalResult(chunks, result.analysis, confidence, result.early_exit, result.standout, result.fusion)
    evidence_found, grounded = select_grounded(gated)
    return SearchResult(
        chunks=grounded,
        analysis=result.analysis,
        confidence=round(confidence, 3),
        early_exit=result.early_exit,
        evidence_found=evidence_found,
        reranked=reranked,
        timings_ms=timings,
        plan=plan,
    )
