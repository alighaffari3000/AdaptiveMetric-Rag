"""Shared machinery for the retrieval evaluation: corpus, judging, metrics.

Relevance is judged by content, never by chunk id, so the same golden file
keeps working when chunking, embeddings, or ranking change in later phases.
"""

from __future__ import annotations

import asyncio
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


async def build_corpus(settings) -> list[dict[str, Any]]:
    """Ingest every corpus file into the (scratch) database."""
    from app import database
    from app.documents import ingest

    pdf = CORPUS_DIR / "annual-review.en.pdf"
    if not pdf.exists():
        from .make_fixtures import main as make_fixtures

        make_fixtures()

    database.init_db()
    database.execute("DELETE FROM documents")
    ingested = []
    for path in corpus_files():
        ingested.append(
            await ingest(path.name, "", path.read_bytes(), settings.chunk_size, settings.chunk_overlap, settings)
        )
    return ingested


def all_chunks() -> list[dict[str, Any]]:
    from app import database

    return database.rows(
        "SELECT c.id,c.content,c.page,c.section,d.name document_name "
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
            if case.doc or case.must_contain or case.any_contain:
                problems.append(f"{case.id}: expect_abstain cases must not declare a target document")
            continue
        if not case.doc:
            problems.append(f"{case.id}: missing 'doc'")
            continue
        if case.doc not in names:
            problems.append(f"{case.id}: document {case.doc!r} is not in the corpus")
            continue
        hits = sum(1 for chunk in chunks if case.matches(chunk))
        if hits == 0:
            if case.blocked_by:
                blocked.append(f"{case.id}: unsatisfiable until {case.blocked_by} ({case.note or 'no note'})")
            else:
                problems.append(f"{case.id}: no chunk satisfies must_contain/any_contain")
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
    return {
        "hit@5": 1.0 if found else 0.0,
        "recall@5": found / total_relevant if total_relevant else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "ndcg@10": (_dcg(flags[:k_ndcg]) / ideal) if ideal else 0.0,
        "first_rank": float(first),
    }


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

    summary = {
        "cases": len(outcomes),
        "answerable": len(answerable),
        "abstain_cases": len(abstaining),
        "hit@5": mean([o.metrics["hit@5"] for o in answerable]),
        "recall@5": mean([o.metrics["recall@5"] for o in answerable]),
        "mrr": mean([o.metrics["mrr"] for o in answerable]),
        "ndcg@10": mean([o.metrics["ndcg@10"] for o in answerable]),
        "answerable_not_abstained": mean([0.0 if o.abstained else 1.0 for o in answerable]),
        "abstain_accuracy": mean([1.0 if o.abstain_correct else 0.0 for o in abstaining]),
        "latency_p50_ms": pct(latencies, .50),
        "latency_p95_ms": pct(latencies, .95),
    }
    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({tag for o in answerable for tag in o.case.tags}):
        tagged = [o for o in answerable if tag in o.case.tags]
        by_tag[tag] = {
            "cases": len(tagged),
            "hit@5": mean([o.metrics["hit@5"] for o in tagged]),
            "recall@5": mean([o.metrics["recall@5"] for o in tagged]),
            "mrr": mean([o.metrics["mrr"] for o in tagged]),
            "ndcg@10": mean([o.metrics["ndcg@10"] for o in tagged]),
        }
    return {"summary": summary, "by_tag": by_tag}


async def run_cases(cases: list[GoldenCase], settings, candidate_count: int | None = None,
                    context_count: int = 10) -> list[CaseOutcome]:
    from app.embeddings import create_query_embedding
    from app.retrieval import retrieve, select_grounded

    chunks = all_chunks()
    outcomes: list[CaseOutcome] = []
    for case in cases:
        total_relevant = sum(1 for chunk in chunks if case.matches(chunk))
        started = time.perf_counter()
        vector = await create_query_embedding(settings, case.question)
        result = retrieve(
            case.question,
            candidate_count or settings.candidate_count,
            context_count,
            None,
            vector,
        )
        latency = (time.perf_counter() - started) * 1000
        evidence_found, _ = select_grounded(result)
        abstained = not evidence_found
        outcomes.append(
            CaseOutcome(
                case=case,
                metrics=score_ranking(case, result.chunks, total_relevant),
                abstained=abstained,
                abstain_correct=(abstained == case.expect_abstain),
                latency_ms=latency,
                top_document=result.chunks[0]["document_name"] if result.chunks else "",
                total_relevant=total_relevant,
            )
        )
    return outcomes


def cleanup(data_dir: Path) -> None:
    if os.environ.get("EVAL_KEEP_DATA"):
        return
    shutil.rmtree(data_dir, ignore_errors=True)
