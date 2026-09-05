"""How good is the claim checker, measured without a generation provider.

The phase's acceptance criterion is a share of cited claims that the cited
passage really supports. Measuring that on model-written answers needs a model;
what can be measured offline is the checker itself, on two populations built
from the golden corpus:

- positives: the extractive answer for each question, whose sentences are
  quotations of the passage they cite. Every one is supported by construction,
  so anything the checker rejects here is a false alarm.
- negatives: the same sentences re-cited to a passage from another document.
  None is supported, so anything the checker accepts here is a miss.

Run:  python -m eval.verify_claims
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .harness import EmbeddingCache, build_corpus, cleanup, load_golden, use_scratch_data_dir

DATA_DIR = use_scratch_data_dir()  # must run before any app import


async def answer_for(case, settings, cache, windows, context_count):
    """The context and extractive answer this question would produce today."""
    from app.embeddings import create_query_embedding, is_semantic
    from app.retrieval import retrieve
    from app.rewrite import rule_plan
    from app.providers import local_answer

    plan = rule_plan(case.question, case.history, settings.history_turns)
    if is_semantic(settings) and cache.enabled:
        vector = (await cache.embed(settings, [plan.question], kind="query"))[0]
    else:
        vector = await create_query_embedding(settings, plan.question)
    semantic = is_semantic(settings)
    result = retrieve(plan.question, settings.candidate_count, context_count, None, vector,
                      settings.semantic_dense_weight if semantic else None,
                      "rank" if semantic else "linear")

    context, seen = [], set()
    for chunk in result.chunks[:settings.context_count]:
        parent = chunk.get("parent_id")
        if parent and parent in seen:
            continue
        seen.add(parent)
        context.append({"document_name": chunk["document_name"], "document_id": chunk["document_id"],
                        "content": windows.get(parent or "", chunk["content"])})
    if not context:
        return [], ""
    return context, local_answer(case.question, context, case.language)


async def main_async(args: argparse.Namespace) -> int:
    from app import database
    from app.claims import claims_from_markers, strip_markers
    from app.main import load_settings
    from app.verify import check_claim

    database.init_db()
    settings = load_settings()
    cases = [case for case in load_golden() if not case.expect_abstain]
    cache = EmbeddingCache(settings, enabled=True)
    await build_corpus(settings, cache)

    windows = {row["id"]: row["content"] for row in database.rows("SELECT id,content FROM parents")}
    by_document: dict[str, list[str]] = {}
    for row in database.rows("SELECT document_id,content FROM chunks"):
        by_document.setdefault(row["document_id"], []).append(row["content"])

    positives = far = near = false_alarms = far_misses = near_misses = uncited = 0
    examples: list[str] = []
    for case in cases:
        context, answer = await answer_for(case, settings, cache, windows, args.context)
        if not context:
            continue
        sources = {number: chunk["content"] for number, chunk in enumerate(context, 1)}
        used = {chunk["document_id"] for chunk in context}
        unrelated = next((contents[0] for document_id, contents in by_document.items()
                          if document_id not in used), "")
        siblings = by_document.get(context[0]["document_id"], [])
        for claim in claims_from_markers(answer):
            if not claim.source_ids:
                uncited += 1
                continue
            # The harder control: another passage of the same document, which
            # shares its vocabulary, its names and often its numbers - and which
            # must not happen to contain the claim, or it is not a negative at
            # all. Chunks of one document overlap by design, so this has to be
            # checked rather than assumed.
            statement = strip_markers(claim.text)
            sibling = next((content for content in siblings if statement not in content), "")
            positives += 1
            if check_claim(claim, sources) is False:
                false_alarms += 1
                if len(examples) < 5:
                    examples.append(claim.text[:110])
            if unrelated:
                far += 1
                if check_claim(claim, {number: unrelated for number in claim.source_ids}) is not False:
                    far_misses += 1
            if sibling:
                near += 1
                if check_claim(claim, {number: sibling for number in claim.source_ids}) is not False:
                    near_misses += 1

    report = {
        "cited_claims": positives,
        "supported_share": round(1 - false_alarms / positives, 4) if positives else 0.0,
        "false_alarms": false_alarms,
        "far_negatives": far,
        "far_rejected_share": round(1 - far_misses / far, 4) if far else 0.0,
        "near_negatives": near,
        "near_rejected_share": round(1 - near_misses / near, 4) if near else 0.0,
        "claims_without_a_citation": uncited,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if examples:
        print("\nfalse alarms (supported claims the checker rejected):")
        for text in examples:
            print(f"  - {text}")
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {target}")
    cache.save()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the claim checker on the golden corpus")
    parser.add_argument("--context", type=int, default=10, help="ranked chunks retrieved per question")
    parser.add_argument("--json", help="write the report to this path")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    finally:
        cleanup(DATA_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
