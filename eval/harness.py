"""Shared machinery for the retrieval evaluation: corpus, judging, metrics.

Relevance is judged by content, never by chunk id, so the same golden file
keeps working when chunking, embeddings, or ranking change in later phases.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .normalize import normalize

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_PATH = EVAL_DIR / "golden.jsonl"
CACHE_DIR = EVAL_DIR / "results" / "embedding-cache"


class EmbeddingCache:
    """Persist embeddings between runs so tuning does not re-pay for API calls.

    Keyed by provider, model, task kind and the exact text, so a cache entry can
    never be served for a different model or a different query.
    """

    def __init__(self, settings, enabled: bool = True):
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        slug = f"{settings.embedding_provider}--{settings.embedding_model}".replace("/", "_").replace(":", "_")
        self.path = CACHE_DIR / f"{slug}.json"
        self.store: dict[str, list[float]] = {}
        if self.enabled and self.path.exists():
            self.store = json.loads(self.path.read_text(encoding="utf-8"))

    @staticmethod
    def _key(kind: str, text: str) -> str:
        return f"{kind}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    async def embed(self, settings, texts: list[str], kind: str = "document") -> list[list[float]]:
        from app.embeddings import create_embeddings

        if not self.enabled:
            return await create_embeddings(settings, texts, kind=kind)
        keys = [self._key(kind, text) for text in texts]
        missing = [text for text, key in zip(texts, keys) if key not in self.store]
        if missing:
            fresh = await create_embeddings(settings, missing, kind=kind)
            for text, vector in zip(missing, fresh):
                self.store[self._key(kind, text)] = vector
            self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        return [self.store[key] for key in keys]

    def save(self) -> None:
        if not self.enabled or not self.store:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.store), encoding="utf-8")


def use_scratch_data_dir() -> Path:
    """Point the application at a throwaway database before app modules load."""
    existing = os.environ.get("EVAL_DATA_DIR")
    target = Path(existing) if existing else Path(tempfile.mkdtemp(prefix="amr-eval-"))
    os.environ["DATA_DIR"] = str(target)
    return target


@dataclass
class GoldenCase:
    id: str
    question: str
    language: str
    tags: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    doc: str | None = None
    must_contain: list[str] = field(default_factory=list)
    any_contain: list[str] = field(default_factory=list)
    relevant_section: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    expect_abstain: bool = False
    blocked_by: str = ""
    note: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "GoldenCase":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"golden case {payload.get('id')!r} has unknown fields: {sorted(unknown)}")
        return cls(**payload)

    def matches(self, chunk: dict[str, Any]) -> bool:
        """A chunk is relevant when it is from the right document and carries the answer."""
        if self.doc and chunk.get("document_name") != self.doc:
            return False
        content = normalize(chunk.get("content", ""))
        if self.must_contain and not all(normalize(item) in content for item in self.must_contain):
            return False
        if self.any_contain and not any(normalize(item) in content for item in self.any_contain):
            return False
        return bool(self.must_contain or self.any_contain or self.doc)

    def section_matches(self, chunk: dict[str, Any]) -> bool:
        """Whether a chunk sits under the section this case expects.

        Judged by the section title's text, not by any identifier, for the same
        reason `matches` is: the golden file has to survive a change of chunker.
        A case that declares no section never matches, so section metrics stay
        confined to the cases that opted into them.
        """
        if not self.relevant_section:
            return False
        if self.doc and chunk.get("document_name") != self.doc:
            return False
        path = chunk.get("section_path") or chunk.get("section") or ""
        return normalize(self.relevant_section) in normalize(path)


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number} is not valid JSON: {exc}") from exc
        case = GoldenCase.from_json(payload)
        if case.id in seen:
            raise ValueError(f"{path.name}:{number} duplicates case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def corpus_files() -> list[Path]:
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"corpus directory is missing: {CORPUS_DIR}")
    return sorted(p for p in CORPUS_DIR.iterdir() if p.is_file() and not p.name.startswith("."))


async def build_corpus(settings, cache: EmbeddingCache | None = None) -> list[dict[str, Any]]:
    """Ingest every corpus file into the (scratch) database."""
    from app import database
    from app.documents import ingest

    pdf = CORPUS_DIR / "annual-review.en.pdf"
    if not pdf.exists():
        from .make_fixtures import main as make_fixtures

        make_fixtures()

    database.init_db()
    database.execute("DELETE FROM documents")
    if cache is not None and cache.enabled:
        import app.documents as documents_module

        original = documents_module.create_embeddings

        async def cached(settings_, texts, kind="document"):
            return await cache.embed(settings_, texts, kind=kind)

        documents_module.create_embeddings = cached
    try:
        ingested = []
        for path in corpus_files():
            ingested.append(
                await ingest(path.name, "", path.read_bytes(),
                             settings.chunk_size, settings.chunk_overlap, settings)
            )
    finally:
        if cache is not None and cache.enabled:
            documents_module.create_embeddings = original
            cache.save()
    from app.index import index as chunk_index

    chunk_index.invalidate()
    return ingested


def all_chunks() -> list[dict[str, Any]]:
    from app import database

    return database.rows(
        "SELECT c.id,c.content,c.page,c.section,c.section_path,c.metadata,d.name document_name "
        "FROM chunks c JOIN documents d ON d.id=c.document_id"
    )


def validate_golden(cases: Iterable[GoldenCase]) -> tuple[list[str], list[str]]:
    """Split golden cases into real inconsistencies and known-blocked expectations.

    A case carrying `blocked_by` is expected to have no satisfying chunk until
    that phase lands; it still scores zero, it just is not a golden-set bug.
    """
    chunks = all_chunks()
    names = {chunk["document_name"] for chunk in chunks}
    problems: list[str] = []
    blocked: list[str] = []
    for case in cases:
        if case.expect_abstain:
            if case.doc or case.must_contain or case.any_contain or case.relevant_section:
                problems.append(f"{case.id}: expect_abstain cases must not declare a target document")
            continue
        if not case.doc:
            problems.append(f"{case.id}: missing 'doc'")
            continue
        if case.doc not in names:
            problems.append(f"{case.id}: document {case.doc!r} is not in the corpus")
            continue
        unmet: list[str] = []
        if not any(case.matches(chunk) for chunk in chunks):
            unmet.append("no chunk satisfies must_contain/any_contain")
        if case.relevant_section and not any(case.section_matches(chunk) for chunk in chunks):
            unmet.append(f"no chunk sits under section {case.relevant_section!r}")
        if unmet:
            detail = "; ".join(unmet)
            if case.blocked_by:
                blocked.append(f"{case.id}: {detail} — until {case.blocked_by} ({case.note or 'no note'})")
            else:
                problems.append(f"{case.id}: {detail}")
        elif case.blocked_by:
            blocked.append(f"{case.id}: marked blocked_by={case.blocked_by} but the corpus already satisfies it")
    return problems, blocked


def _dcg(gains: list[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def score_ranking(case: GoldenCase, ranked: list[dict[str, Any]], total_relevant: int,
                  k_recall: int = 5, k_ndcg: int = 10) -> dict[str, float]:
    flags = [1 if case.matches(chunk) else 0 for chunk in ranked]
    found = sum(flags[:k_recall])
    first = next((i + 1 for i, flag in enumerate(flags) if flag), 0)
    ideal = _dcg([1] * min(total_relevant, k_ndcg))
    metrics = {
        "hit@5": 1.0 if found else 0.0,
        "recall@5": found / total_relevant if total_relevant else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "ndcg@10": (_dcg(flags[:k_ndcg]) / ideal) if ideal else 0.0,
        "first_rank": float(first),
    }
    if case.relevant_section:
        # Whether retrieval landed in the right part of the document, which is
        # what a structure signal is supposed to move. Kept separate from the
        # content metrics: a chunk can carry the answer text and still be filed
        # under the wrong heading, and only this pair notices.
        section_flags = [1 if case.section_matches(chunk) else 0 for chunk in ranked]
        metrics["section_hit@1"] = float(section_flags[0]) if section_flags else 0.0
        metrics["section_recall@3"] = 1.0 if any(section_flags[:3]) else 0.0
    return metrics


@dataclass
class CaseOutcome:
    case: GoldenCase
    metrics: dict[str, float]
    abstained: bool
    abstain_correct: bool
    latency_ms: float
    top_document: str
    total_relevant: int


def aggregate(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    answerable = [o for o in outcomes if not o.case.expect_abstain]
    abstaining = [o for o in outcomes if o.case.expect_abstain]
    latencies = sorted(o.latency_ms for o in outcomes)

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
        return round(values[index], 1)

    structural = [o for o in answerable if o.case.relevant_section]
    summary = {
        "cases": len(outcomes),
        "answerable": len(answerable),
        "abstain_cases": len(abstaining),
        "section_cases": len(structural),
        "hit@5": mean([o.metrics["hit@5"] for o in answerable]),
        "recall@5": mean([o.metrics["recall@5"] for o in answerable]),
        "mrr": mean([o.metrics["mrr"] for o in answerable]),
        "ndcg@10": mean([o.metrics["ndcg@10"] for o in answerable]),
        "answerable_not_abstained": mean([0.0 if o.abstained else 1.0 for o in answerable]),
        "abstain_accuracy": mean([1.0 if o.abstain_correct else 0.0 for o in abstaining]),
        "section_hit@1": mean([o.metrics.get("section_hit@1", 0.0) for o in structural]),
        "section_recall@3": mean([o.metrics.get("section_recall@3", 0.0) for o in structural]),
        "latency_p50_ms": pct(latencies, .50),
        "latency_p95_ms": pct(latencies, .95),
    }
    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({tag for o in answerable for tag in o.case.tags}):
        tagged = [o for o in answerable if tag in o.case.tags]
        sectioned = [o for o in tagged if o.case.relevant_section]
        by_tag[tag] = {
            "cases": len(tagged),
            "hit@5": mean([o.metrics["hit@5"] for o in tagged]),
            "recall@5": mean([o.metrics["recall@5"] for o in tagged]),
            "mrr": mean([o.metrics["mrr"] for o in tagged]),
            "ndcg@10": mean([o.metrics["ndcg@10"] for o in tagged]),
        }
        if sectioned:
            by_tag[tag]["section_cases"] = len(sectioned)
            by_tag[tag]["section_hit@1"] = mean([o.metrics.get("section_hit@1", 0.0) for o in sectioned])
            by_tag[tag]["section_recall@3"] = mean([o.metrics.get("section_recall@3", 0.0) for o in sectioned])
    return {"summary": summary, "by_tag": by_tag}


async def run_cases(cases: list[GoldenCase], settings, candidate_count: int | None = None,
                    context_count: int = 10, cache: EmbeddingCache | None = None) -> list[CaseOutcome]:
    """Score the full pipeline: embed, fuse, rerank when enabled, and gate."""
    from app.embeddings import create_query_embedding, is_semantic
    from app.rerank import blend, rerank_scores
    from app.retrieval import retrieve, score_confidence, select_grounded, RetrievalResult

    async def embed_query(question: str) -> list[float]:
        if cache is not None and cache.enabled and is_semantic(settings):
            return (await cache.embed(settings, [question], kind="query"))[0]
        return await create_query_embedding(settings, question)

    semantic = is_semantic(settings)
    dense_weight = settings.semantic_dense_weight if semantic else None
    fusion = "rank" if semantic else "linear"
    depth = max(context_count, settings.rerank_top_n) if settings.rerank_enabled else context_count

    chunks = all_chunks()
    outcomes: list[CaseOutcome] = []
    for case in cases:
        total_relevant = sum(1 for chunk in chunks if case.matches(chunk))
        started = time.perf_counter()
        vector = await embed_query(case.question)
        result = retrieve(case.question, candidate_count or settings.candidate_count, depth,
                          None, vector, dense_weight, fusion)
        ranked = result.chunks
        confidence = result.confidence
        if settings.rerank_enabled and len(ranked) > 1:
            shortlist = ranked[:settings.rerank_top_n]
            try:
                scores = await rerank_scores(settings, case.question, shortlist)
                ranked = blend(shortlist, scores, settings.rerank_weight) + ranked[settings.rerank_top_n:]
                confidence = score_confidence(ranked[:context_count],
                                              result.analysis.query_tokens or result.analysis.keywords,
                                              result.standout)
            except Exception as exc:
                # A silently skipped reranker would be reported as a measurement
                # of reranking, which it is not.
                raise RuntimeError(
                    f"reranking failed on case {case.id!r}: {type(exc).__name__}: {exc}"
                ) from exc
        latency = (time.perf_counter() - started) * 1000
        ranked = ranked[:context_count]
        gated = RetrievalResult(ranked, result.analysis, confidence, result.early_exit, result.standout,
                                result.fusion)
        evidence_found, _ = select_grounded(gated)
        abstained = not evidence_found
        outcomes.append(
            CaseOutcome(
                case=case,
                metrics=score_ranking(case, ranked, total_relevant),
                abstained=abstained,
                abstain_correct=(abstained == case.expect_abstain),
                latency_ms=latency,
                top_document=ranked[0]["document_name"] if ranked else "",
                total_relevant=total_relevant,
            )
        )
    if cache is not None:
        cache.save()
    return outcomes


def cleanup(data_dir: Path) -> None:
    if os.environ.get("EVAL_KEEP_DATA"):
        return
    shutil.rmtree(data_dir, ignore_errors=True)
