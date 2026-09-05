"""Measure retrieval latency as the chunk count grows.

Run:  python -m eval.bench_latency
      python -m eval.bench_latency --sizes 1000 5000 20000 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import uuid
from pathlib import Path

from .harness import cleanup, use_scratch_data_dir

DATA_DIR = use_scratch_data_dir()  # must run before any app import

FA_WORDS = (
    "قرارداد مبلغ تاریخ پایان شرکت مدیر پرداخت ریال میلیون سال ماه فروردین اردیبهشت خرداد "
    "تمدید فسخ تعهد طرف اول دوم مالیات بیمه کارفرما پیمانکار تحویل خدمات پشتیبانی زیرساخت"
).split()
EN_WORDS = (
    "contract amount date expiry company manager payment invoice million year month renewal "
    "termination obligation party tax insurance employer contractor delivery support platform"
).split()

QUERIES = [
    "قرارداد شماره 137 چه زمانی تمام می‌شود؟",
    "مبلغ کل قرارداد چقدر است؟",
    "What is the renewal notice period?",
    "چرا قرارداد فسخ شد؟",
]


def seed_chunks(count: int) -> None:
    from app import database
    from app.retrieval import embed, tokenize

    database.init_db()
    database.execute("DELETE FROM documents")
    document_id = uuid.uuid4().hex
    now = "2026-01-01T00:00:00+00:00"
    rng = random.Random(1234)
    with database.connect() as db:
        db.execute(
            "INSERT INTO documents(id,name,type,size,chunks,created_at,metadata) VALUES(?,?,?,?,?,?,?)",
            (document_id, "bench.txt", "text/plain", 0, count, now, "{}"),
        )
        rows = []
        for position in range(count):
            words = [rng.choice(FA_WORDS if rng.random() < .7 else EN_WORDS) for _ in range(120)]
            text = " ".join(words) + f" شماره {rng.randint(1, 999)} سال {rng.randint(1395, 1405)}."
            rows.append((
                uuid.uuid4().hex, document_id, position, 1, None, text,
                database.json_value(embed(text)), database.json_value(tokenize(text)), "{}",
            ))
        db.executemany(
            "INSERT INTO chunks(id,document_id,position,page,section,content,embedding,tokens,metadata) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            rows,
        )


def measure(size: int, repeats: int) -> dict:
    from app.retrieval import retrieve

    seed_chunks(size)
    retrieve(QUERIES[0])  # warm caches
    timings: list[float] = []
    for index in range(repeats * len(QUERIES)):
        query = QUERIES[index % len(QUERIES)]
        started = time.perf_counter()
        retrieve(query)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {
        "chunks": size,
        "runs": len(timings),
        "mean_ms": round(statistics.fmean(timings), 1),
        "p50_ms": round(timings[len(timings) // 2], 1),
        "p95_ms": round(timings[min(len(timings) - 1, int(len(timings) * .95))], 1),
        "max_ms": round(timings[-1], 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency by corpus size")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
    parser.add_argument("--repeats", type=int, default=3, help="passes over the query set (default 3)")
    parser.add_argument("--json", help="write results to this path")
    args = parser.parse_args()

    try:
        results = []
        print(f"  {'chunks':>8}{'runs':>6}{'mean':>10}{'p50':>10}{'p95':>10}{'max':>10}")
        for size in args.sizes:
            row = measure(size, args.repeats)
            results.append(row)
            print(f"  {row['chunks']:>8}{row['runs']:>6}{row['mean_ms']:>9.1f}m"
                  f"{row['p50_ms']:>9.1f}m{row['p95_ms']:>9.1f}m{row['max_ms']:>9.1f}m")
        if args.json:
            target = Path(args.json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nwrote {target}")
        return 0
    finally:
        cleanup(DATA_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
