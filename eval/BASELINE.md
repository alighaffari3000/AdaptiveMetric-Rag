# Retrieval baseline — before Phase 1

Recorded 2026-09-05 on the state tagged by commit "docs: add phased architecture
improvement plan". Every later phase is compared against these numbers.

Reproduce with:

```
python -m eval.run_eval --json eval/results/baseline.json --markdown eval/results/baseline.md
python -m eval.bench_latency
```

## Configuration

| Setting | Value |
|---|---|
| Embedding | `local` / `multilingual-feature-hashing-v1` (384-dim feature hashing) |
| Candidate pool | 100 |
| Chunk size / overlap | 900 / 140 characters |
| Ranked chunks scored | 10 |
| Corpus | 13 documents, 31 chunks |
| Golden cases | 89 (81 answerable, 8 abstain) |

## Quality

| Metric | Baseline |
|---|---:|
| hit@5 | 0.800 |
| recall@5 | 0.794 |
| MRR | 0.709 |
| nDCG@10 | 0.736 |
| answerable_not_abstained | 0.944 |
| abstain_accuracy | 0.125 |
| retrieval latency p50 | 12.2 ms |
| retrieval latency p95 | 16.1 ms |

## By tag

| Tag | Cases | hit@5 | recall@5 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| browse | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| causal | 14 | 0.714 | 0.714 | 0.631 | 0.643 |
| conceptual | 7 | 0.571 | 0.571 | 0.520 | 0.566 |
| cross_lingual | 15 | 0.400 | 0.400 | 0.192 | 0.273 |
| distractor_pressure | 4 | 1.000 | 1.000 | 1.000 | 0.980 |
| en2en | 27 | 1.000 | 1.000 | 0.957 | 0.957 |
| en2fa | 5 | 0.600 | 0.600 | 0.233 | 0.326 |
| entity | 10 | 1.000 | 1.000 | 0.900 | 0.926 |
| fa2en | 10 | 0.300 | 0.300 | 0.172 | 0.246 |
| fa2fa | 43 | 0.837 | 0.826 | 0.747 | 0.773 |
| factual | 14 | 0.929 | 0.929 | 0.829 | 0.870 |
| follow_up | 4 | 0.500 | 0.500 | 0.375 | 0.408 |
| multi_intent | 8 | 0.750 | 0.750 | 0.667 | 0.677 |
| numeric | 31 | 0.742 | 0.726 | 0.651 | 0.675 |
| page_boundary | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| pdf | 6 | 0.500 | 0.500 | 0.528 | 0.559 |
| technical | 8 | 1.000 | 1.000 | 0.938 | 0.954 |
| temporal | 11 | 1.000 | 1.000 | 0.826 | 0.869 |

## Latency by corpus size

Full `retrieve()` call, synthetic corpus, 12 runs per size.

| Chunks | mean | p50 | p95 |
|---:|---:|---:|---:|
| 1,000 | 272.7 ms | 257.2 ms | 384.9 ms |
| 5,000 | 1,317.2 ms | 1,275.3 ms | 1,560.9 ms |
| 20,000 | 6,556.1 ms | 6,471.1 ms | 8,305.5 ms |

Latency grows linearly with the corpus: every query decodes every stored vector
from JSON, scores it in pure Python, and rebuilds the BM25 statistics from
scratch. Phase 1 targets under 50 ms at 20,000 chunks.

## What the baseline says

**Cross-lingual retrieval is the largest quality gap.** A Persian question against
an English document reaches hit@5 of 0.300. The default embedding is feature
hashing over tokens, so a Persian query and its English answer share no features;
the only bridge is the 20-entry hand-written keyword dictionary in
`app/retrieval.py`.

**Abstention barely works.** The gate accepts 7 of 8 unanswerable questions, and
it wrongly abstains on 3 of 4 conversational follow-ups. The confidence value is
not calibrated against anything.

**English monolingual retrieval already saturates.** At 1.000 hit@5 this slice
cannot show improvement; it exists to catch regressions, not progress.

**Two cases are deliberately unsatisfiable.** The page-boundary cases in
`annual-review.en.pdf` have no chunk that can match, because chunking never
crosses a PDF page break. They score zero until Phase 5 and are marked
`blocked_by: phase-5` in the golden file.

---

# Phase 1 — index and performance

Recorded 2026-09-05 after replacing the per-query full scan with an in-memory
index. This phase is deliberately quality-neutral: it changes where the work
happens, not what is computed.

## Quality: unchanged, as intended

| Metric | Baseline | Phase 1 | Delta |
|---|---:|---:|---:|
| hit@5 | 0.800 | 0.800 | 0.000 |
| recall@5 | 0.794 | 0.794 | 0.000 |
| MRR | 0.709 | 0.709 | 0.000 |
| nDCG@10 | 0.736 | 0.736 | 0.000 |
| abstain_accuracy | 0.125 | 0.125 | 0.000 |

Every per-tag figure is identical too. The BM25 formula is reproduced exactly by
the inverted index, and dense scoring is the same clamped cosine expressed as a
matrix product, so scores are bit-for-bit the same.

## Latency: the point of the phase

| Chunks | Baseline mean | Phase 1 mean | Speed-up |
|---:|---:|---:|---:|
| 1,000 | 272.7 ms | 6.5 ms | 42x |
| 5,000 | 1,317.2 ms | 6.9 ms | 191x |
| 20,000 | 6,556.1 ms | 12.7 ms | 516x |

Golden-set retrieval latency fell from 12.2 ms to 2.4 ms at p50.

Acceptance target was under 50 ms at 20,000 chunks. Met with room to spare, and
latency now grows sub-linearly: the matrix product is vectorised and the lexical
pass only touches postings for terms the query actually contains.

## Deviation from the plan

The plan proposed SQLite FTS5 for BM25. An in-process inverted index was used
instead, for a reason that matters at this point in the sequence: FTS5 has its
own tokenizer, so it would have changed BM25 scores and made this phase a
quality change rather than a pure performance change. The inverted index reuses
the tokens already stored per chunk and reproduces the previous formula exactly,
which is what let the table above show zeros. The plan's own stated fallback
("keep the current BM25 but precompute doc_freq and avg_len") pointed the same
way. FTS5 remains available later, once RRF fusion in Phase 3 makes the exact
lexical scale irrelevant.

---

# Phase 2 — semantic embeddings

Recorded 2026-09-05. Measured with `gemini-embedding-001` at 768 dimensions,
since neither Ollama nor a local sentence-transformers install was available on
this machine. The other providers share the same code path.

## The embedding itself is fixed

| Pair | Feature hashing | Gemini |
|---|---:|---:|
| "قرارداد چه زمانی منقضی می‌شود؟" vs "When does the contract expire?" | 0.000 | 0.883 |

## End-to-end quality: improved, but far below what the embedding supports

| Metric | Phase 1 | Phase 2 (Gemini) | Delta |
|---|---:|---:|---:|
| hit@5 | 0.800 | 0.844 | +0.044 |
| recall@5 | 0.794 | 0.833 | +0.039 |
| MRR | 0.709 | 0.735 | +0.026 |
| nDCG@10 | 0.736 | 0.767 | +0.031 |
| answerable_not_abstained | 0.944 | 1.000 | +0.056 |
| cross_lingual hit@5 | 0.400 | 0.533 | +0.133 |
| fa2en hit@5 | 0.300 | 0.400 | +0.100 |
| follow_up hit@5 | 0.500 | 1.000 | +0.500 |

Phase 2's acceptance criteria were cross-lingual recall above 0.6 and overall
recall 15 points above baseline. **Neither is met.** The reason is not the
embedding.

## Diagnosis: the adaptive linear sum suppresses the semantic signal

Ranking the same corpus by the dense score alone, with everything else switched
off, gives:

| Tag | Full adaptive scoring | Dense only |
|---|---:|---:|
| cross_lingual | 0.533 | **1.000** |
| fa2en | 0.400 | **1.000** |
| en2fa | 0.800 | **1.000** |
| conceptual | 0.571 | **1.000** |
| causal | 0.857 | 0.929 |
| numeric | 0.806 | 0.935 |
| fa2fa | 0.884 | 0.977 |

The embedding already ranks every cross-lingual case correctly. The final score
then throws that away, for two compounding reasons:

1. **The dense weight is small for most intents.** Factual, numeric and temporal
   questions weight `dense` at 0.15 to 0.25, and those intents cover most of the
   golden set. The lexical features carry the rest.
2. **The signals are not on comparable scales.** Cosine similarities from a
   semantic model sit in a narrow high band, so the spread between a relevant and
   an irrelevant chunk is perhaps 0.15. BM25 is min-max normalised to fill [0, 1].
   After weighting, a 0.15 dense spread times 0.16 is 0.024, against a lexical
   spread of up to 0.19. The lexical term decides the ranking almost every time.

Adding a better embedding to a weighted sum of unnormalised heterogeneous scores
does not help much, because the sum was never able to use it. This is precisely
what rank-based fusion fixes, so the acceptance criteria for this phase are
carried into Phase 3 rather than declared met here.

## Also in this phase

- Query embedding no longer blends keyword-expanded variants when the provider is
  semantic; blending was a crutch for feature hashing and only blurs a real model.
- Confidence coverage is computed from the words the user actually typed. Counting
  machine-added English expansions as "not covered" penalised every Persian
  question asked against Persian sources.
- Embedding requests retry transient failures with backoff. A single dropped
  connection previously aborted an entire upload or re-index.
- Uploading a file already in the library is rejected by SHA-256 content digest
  instead of silently creating a second copy of every chunk.
