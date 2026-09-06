"""Score the retrieval stack against the golden set.

Run:  python -m eval.run_eval
      python -m eval.run_eval --json eval/results/run.json --markdown eval/results/run.md
      python -m eval.run_eval --tag cross_lingual --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .harness import (
    EmbeddingCache,
    aggregate,
    build_corpus,
    cleanup,
    load_golden,
    run_cases,
    use_scratch_data_dir,
    validate_golden,
)

DATA_DIR = use_scratch_data_dir()  # must run before any app import


def _fmt(value: float) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def to_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Embedding: `{report['runtime']['embedding_provider']}` / `{report['runtime']['embedding_model']}`  ",
        f"Cases: {summary['cases']} ({summary['answerable']} answerable, {summary['abstain_cases']} abstain)",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("hit@5", "recall@5", "mrr", "ndcg@10", "answerable_not_abstained",
                "abstain_accuracy", "latency_p50_ms", "latency_p95_ms"):
        lines.append(f"| {key} | {_fmt(summary[key])} |")
    lines += ["", "## By tag", "", "| Tag | Cases | hit@5 | recall@5 | MRR | nDCG@10 |", "|---|---:|---:|---:|---:|---:|"]
    for tag, stats in report["by_tag"].items():
        lines.append(
            f"| {tag} | {stats['cases']} | {_fmt(stats['hit@5'])} | {_fmt(stats['recall@5'])} "
            f"| {_fmt(stats['mrr'])} | {_fmt(stats['ndcg@10'])} |"
        )
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> int:
    from app import database
    from app.main import load_settings

    database.init_db()
    settings = load_settings()
    if args.embedding:
        settings = settings.model_copy(update={
            "embedding_provider": args.embedding,
            "embedding_model": args.embedding_model or settings.embedding_model,
            "embedding_base_url": args.embedding_base_url or "",
        })
    if args.provider:
        import os

        key = settings.api_key or os.getenv(f"{args.provider.upper()}_API_KEY", "")
        settings = settings.model_copy(update={"provider": args.provider,
                                               "model": args.model or settings.model,
                                               "api_key": key})
    if args.chunk_size or args.chunk_overlap is not None:
        size = args.chunk_size or settings.chunk_size
        overlap = settings.chunk_overlap if args.chunk_overlap is None else args.chunk_overlap
        settings = settings.model_copy(update={"chunk_size": size,
                                               "chunk_overlap": min(overlap, size - 1)})
    if args.flat_chunking:
        import app.documents as documents_module

        structural = documents_module.extract

        def flat(filename, payload):
            # One block per page, structure discarded: what the chunker did
            # before section paths existed.
            merged: dict[int | None, list[str]] = {}
            for text, page, _ in structural(filename, payload):
                merged.setdefault(page, []).append(text)
            return [("\n".join(parts), page, None) for page, parts in merged.items()]

        documents_module.extract = flat
    if args.rerank:
        settings = settings.model_copy(update={
            "rerank_enabled": True,
            "rerank_backend": args.rerank,
            "rerank_model": args.rerank_model or "",
        })
    cases = load_golden()
    if args.tag:
        cases = [case for case in cases if args.tag in case.tags]
        if not cases:
            print(f"no golden case carries tag {args.tag!r}")
            return 2

    cache = EmbeddingCache(settings, enabled=not args.no_cache)
    documents = await build_corpus(settings, cache)
    problems, blocked = validate_golden(cases)
    if problems:
        print("golden set is inconsistent with the corpus:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    if blocked:
        print("known-blocked cases (scored as failures until the named phase lands):")
        for item in blocked:
            print(f"  - {item}")
        print()

    outcomes = await run_cases(cases, settings, context_count=args.context, cache=cache)
    report = aggregate(outcomes)
    report["runtime"] = {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "candidate_count": settings.candidate_count,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "context_count_evaluated": args.context,
        "rerank": {"enabled": settings.rerank_enabled, "backend": settings.rerank_backend,
                   "model": settings.rerank_model or settings.model,
                   "top_n": settings.rerank_top_n, "weight": settings.rerank_weight}
        if settings.rerank_enabled else None,
        "documents": len(documents),
        "chunks": sum(doc["chunks"] for doc in documents),
    }
    report["cases"] = [
        {
            "id": o.case.id,
            "tags": o.case.tags,
            "expect_abstain": o.case.expect_abstain,
            "abstained": o.abstained,
            "abstain_correct": o.abstain_correct,
            "relevant_chunks": o.total_relevant,
            "top_document": o.top_document,
            "latency_ms": round(o.latency_ms, 1),
            **{k: round(v, 4) for k, v in o.metrics.items()},
        }
        for o in outcomes
    ]

    summary = report["summary"]
    print(f"corpus: {report['runtime']['documents']} documents, {report['runtime']['chunks']} chunks")
    print(f"embedding: {settings.embedding_provider} / {settings.embedding_model}"
          f"{f'  (cache: {cache.hits} hits, {cache.misses} new)' if cache.enabled else ''}")
    print()
    for key in ("hit@5", "recall@5", "mrr", "ndcg@10", "answerable_not_abstained",
                "abstain_accuracy", "latency_p50_ms", "latency_p95_ms"):
        print(f"  {key:<26} {_fmt(summary[key])}")
    print()
    print(f"  {'tag':<18}{'n':>4}{'hit@5':>9}{'recall@5':>10}{'MRR':>8}{'nDCG@10':>9}")
    for tag, stats in report["by_tag"].items():
        print(f"  {tag:<18}{stats['cases']:>4}{stats['hit@5']:>9.3f}"
              f"{stats['recall@5']:>10.3f}{stats['mrr']:>8.3f}{stats['ndcg@10']:>9.3f}")

    if args.verbose:
        print("\n  failing answerable cases (no relevant chunk in top 5):")
        for o in outcomes:
            if not o.case.expect_abstain and not o.metrics["hit@5"]:
                print(f"    {o.case.id:<28} top={o.top_document}")
        wrong = [o for o in outcomes if not o.abstain_correct]
        if wrong:
            print("\n  wrong abstention decisions:")
            for o in wrong:
                want = "abstain" if o.case.expect_abstain else "answer"
                got = "abstain" if o.abstained else "answer"
                print(f"    {o.case.id:<28} expected={want} got={got}")

    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {target}")
    if args.markdown:
        target = Path(args.markdown)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(to_markdown(report), encoding="utf-8")
        print(f"wrote {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against the golden set")
    parser.add_argument("--json", help="write the full report to this path")
    parser.add_argument("--markdown", help="write a Markdown summary to this path")
    parser.add_argument("--tag", help="only run cases carrying this tag")
    parser.add_argument("--context", type=int, default=10, help="ranked chunks to score (default 10)")
    parser.add_argument("--verbose", action="store_true", help="list failing cases")
    parser.add_argument("--chunk-size", type=int,
                        help="override the chunk size, to separate granularity effects from ranking ones")
    parser.add_argument("--chunk-overlap", type=int, help="override the chunk overlap")
    parser.add_argument("--flat-chunking", action="store_true",
                        help="ignore document structure when chunking, for A/B comparison")
    parser.add_argument("--embedding", help="override the embedding provider for this run")
    parser.add_argument("--embedding-model", help="override the embedding model for this run")
    parser.add_argument("--embedding-base-url", help="override the embedding base URL for this run")
    parser.add_argument("--rerank", choices=["llm", "cross-encoder"], help="enable reranking for this run")
    parser.add_argument("--rerank-model", help="model the reranker should use")
    parser.add_argument("--provider", help="override the generation provider (for the llm reranker)")
    parser.add_argument("--model", help="override the generation model")
    parser.add_argument("--no-cache", action="store_true",
                        help="always call the embedding provider instead of reusing cached vectors")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    finally:
        cleanup(DATA_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
