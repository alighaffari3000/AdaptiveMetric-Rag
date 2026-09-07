"""Draft golden-set candidates from the corpus, in the style of SearchTome.

STAIR (arXiv:2609.03874) builds its benchmark by asking a model for questions
a passage answers and taking the passage's own section as the gold label. That
is cheap and it is how this project can grow past the eight negative examples
that `eval/BASELINE.md` names as the reason abstention accuracy sits at 0.125.

Two rules keep the generated records from corrupting the yardstick:

- They are written to `eval/golden.generated.jsonl`, never to `golden.jsonl`.
  A machine-written label is a hypothesis; a golden record is a decision. Move
  a record across by hand, after reading it.
- Every record is validated against the corpus before it is written. A question
  whose `must_contain` no chunk satisfies is a bad question, not a hard one,
  and it is dropped with a counted reason rather than saved.

Run:  python -m eval.make_questions --provider gemini --model gemini-2.5-flash
      python -m eval.make_questions --limit 5 --negatives 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from .harness import GoldenCase, all_chunks, build_corpus, cleanup, use_scratch_data_dir
from .normalize import normalize

DATA_DIR = use_scratch_data_dir()  # must run before any app import

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVAL_DIR / "golden.generated.jsonl"

QUESTIONS_PER_CHUNK = 2

POSITIVE_SYSTEM = (
    "You write evaluation questions for a retrieval system. Given a passage, write questions "
    "that the passage answers on its own, and for each one quote the exact substring of the "
    "passage that contains the answer. Quote it verbatim, including digits and punctuation, "
    "and keep it under twelve words. Write the question in the same language as the passage."
)

NEGATIVE_SYSTEM = (
    "You write evaluation questions for a retrieval system. Given a list of topics a document "
    "library covers, write questions that sound like they belong to that library but that the "
    "library cannot answer, because the specific fact is not in it. Do not ask about anything "
    "listed. Vary the language between Persian and English."
)


def positive_prompt(passage: str, count: int) -> str:
    return (
        f"Passage:\n{passage[:1500]}\n\n"
        f'Reply with JSON only: {{"questions": [{{"question": "...", "answer_quote": "..."}}]}}, '
        f"with {count} entries."
    )


def negative_prompt(topics: list[str], count: int) -> str:
    listing = "\n".join(f"- {topic}" for topic in topics[:60])
    return (
        f"The library covers:\n{listing}\n\n"
        f'Reply with JSON only: {{"questions": ["...", "..."]}}, with {count} entries.'
    )


def parse_json_block(raw: str) -> dict[str, Any]:
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("the model did not return JSON")
    return json.loads(match.group(0))


def slug(text: str, limit: int = 24) -> str:
    cleaned = re.sub(r"[^\w؀-ۿ]+", "-", text.strip().lower())
    return cleaned.strip("-")[:limit] or "case"


def leaf_section(section_path: str) -> str:
    """The most specific heading a chunk sits under, for the gold section label."""
    if not section_path:
        return ""
    last = section_path.split(" > ")[-1]
    return last.split(" | ")[0].strip()


def build_positive(chunk: dict[str, Any], question: str, quote: str, index: int) -> GoldenCase | None:
    """A candidate record, or None when the model's quote is not in the passage."""
    if not question.strip() or not quote.strip():
        return None
    if normalize(quote) not in normalize(chunk["content"]):
        return None
    section = leaf_section(chunk.get("section_path") or chunk.get("section") or "")
    return GoldenCase(
        id=f"gen-{slug(chunk['document_name'])}-{index:03d}",
        question=question.strip(),
        language="fa" if re.search(r"[؀-ۿ]", question) else "en",
        doc=chunk["document_name"],
        must_contain=[quote.strip()],
        relevant_section=section,
        tags=["generated", "structural" if section else "flat"],
        note="machine-drafted; review before moving into golden.jsonl",
    )


def build_negative(question: str, index: int) -> GoldenCase | None:
    if not question.strip():
        return None
    return GoldenCase(
        id=f"gen-abstain-{index:03d}",
        question=question.strip(),
        language="fa" if re.search(r"[؀-ۿ]", question) else "en",
        expect_abstain=True,
        tags=["generated", "abstain"],
        note="machine-drafted; review before moving into golden.jsonl",
    )


def to_json_line(case: GoldenCase) -> str:
    payload: dict[str, Any] = {"id": case.id, "question": case.question, "language": case.language}
    if case.expect_abstain:
        payload["expect_abstain"] = True
    else:
        payload["doc"] = case.doc
        payload["must_contain"] = case.must_contain
        if case.relevant_section:
            payload["relevant_section"] = case.relevant_section
    payload["tags"] = case.tags
    payload["note"] = case.note
    return json.dumps(payload, ensure_ascii=False)


async def main_async(args: argparse.Namespace) -> int:
    import os

    from app import database
    from app.main import load_settings
    from app.providers import complete

    database.init_db()
    settings = load_settings()
    if args.provider:
        key = settings.api_key or os.getenv(f"{args.provider.upper()}_API_KEY", "")
        settings = settings.model_copy(update={"provider": args.provider,
                                               "model": args.model or settings.model,
                                               "api_key": key})
    if settings.provider == "local":
        print("Question generation needs a generation provider; pass --provider.")
        return 2

    await build_corpus(settings)
    chunks = all_chunks()
    if args.limit:
        chunks = chunks[:args.limit]

    drafted: list[GoldenCase] = []
    dropped = {"unquoted": 0, "unparsed": 0, "failed": 0}
    serial = 0  # a single counter, so --per-chunk above 10 cannot collide

    for index, chunk in enumerate(chunks):
        try:
            raw = await complete(settings, POSITIVE_SYSTEM,
                                 positive_prompt(chunk["content"], args.per_chunk),
                                 model=settings.model, temperature=0.2, max_tokens=700)
            payload = parse_json_block(raw)
        except ValueError:
            dropped["unparsed"] += 1
            continue
        except Exception as exc:
            print(f"  chunk {index}: {type(exc).__name__}: {exc}")
            dropped["failed"] += 1
            continue
        entries = payload.get("questions")
        if not isinstance(entries, list):
            dropped["unparsed"] += 1
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                dropped["unparsed"] += 1
                continue
            serial += 1
            case = build_positive(chunk, str(entry.get("question", "")),
                                  str(entry.get("answer_quote", "")), serial)
            if case is None:
                dropped["unquoted"] += 1
                continue
            drafted.append(case)

    if args.negatives:
        topics = sorted({f"{chunk['document_name']}: {leaf_section(chunk.get('section_path') or '')}"
                         for chunk in all_chunks()})
        try:
            payload = parse_json_block(await complete(
                settings, NEGATIVE_SYSTEM, negative_prompt(topics, args.negatives),
                model=settings.model, temperature=0.6, max_tokens=1200))
            for offset, question in enumerate(payload.get("questions") or []):
                case = build_negative(str(question), offset)
                if case is None:
                    dropped["unparsed"] += 1
                    continue
                drafted.append(case)
        except Exception as exc:
            print(f"  negatives: {type(exc).__name__}: {exc}")
            dropped["failed"] += 1

    # Ids come from one monotonic counter, so they cannot collide within a run.
    unique = drafted

    header = [
        "// Machine-drafted candidates. NOT part of the evaluation.",
        "// Every record here is a hypothesis: read it, fix it or delete it, and only",
        "// then move it into golden.jsonl. An unreviewed label makes the yardstick",
        "// agree with the retriever instead of with the corpus.",
    ]
    OUTPUT_PATH.write_text("\n".join(header + [to_json_line(case) for case in unique]) + "\n",
                           encoding="utf-8")
    positives = sum(1 for case in unique if not case.expect_abstain)
    print(f"wrote {OUTPUT_PATH}: {positives} answerable, {len(unique) - positives} abstain")
    print(f"dropped: {dropped['unquoted']} unquoted, {dropped['unparsed']} unparsed, "
          f"{dropped['failed']} failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="generation provider to draft with")
    parser.add_argument("--model", help="model to draft with")
    parser.add_argument("--per-chunk", type=int, default=QUESTIONS_PER_CHUNK,
                        help="questions to ask for per chunk (default 2)")
    parser.add_argument("--negatives", type=int, default=30,
                        help="unanswerable questions to draft (default 30)")
    parser.add_argument("--limit", type=int, help="only use the first N chunks")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    finally:
        cleanup(DATA_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
