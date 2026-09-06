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
| Golden cases | 98 (90 answerable, 8 abstain) |

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

---

# Phase 3 — rank fusion and reranking

Recorded 2026-09-05.

## What changed

**Fusion follows the signal.** The weighted sum survives for the feature-hashing
default, where its hand-tuned weights still measure better than rank fusion;
rank fusion is used whenever the embedding is semantic. The fusion method should
match the nature of the signals, and rank fusion hands a meaningless dense score
a fairer share of the vote than it deserves.

**The dense weight follows the embedding.** With a semantic model the dense
signal is raised to `semantic_dense_weight` (default 0.85) and the remaining
share is distributed among the other signals in their original intent
proportions, so the adaptive idea survives while the strongest signal leads.

## Results with a semantic embedding (gemini-embedding-001, no reranker)

| Metric | Baseline | Phase 2 | Phase 3 | vs baseline |
|---|---:|---:|---:|---:|
| hit@5 | 0.800 | 0.844 | **0.956** | +0.156 |
| recall@5 | 0.794 | 0.833 | **0.944** | +0.150 |
| MRR | 0.709 | 0.735 | **0.775** | +0.066 |
| nDCG@10 | 0.736 | 0.767 | **0.818** | +0.082 |
| cross_lingual hit@5 | 0.400 | 0.533 | **0.933** | +0.533 |
| fa2en hit@5 | 0.300 | 0.400 | **0.900** | +0.600 |
| en2fa hit@5 | 0.600 | 0.800 | **1.000** | +0.400 |
| conceptual hit@5 | 0.571 | 0.571 | **1.000** | +0.429 |
| numeric hit@5 | 0.742 | 0.806 | **0.968** | +0.226 |

The acceptance criteria carried over from Phase 2 are now met: cross-lingual
recall is 0.900 against a target of 0.6, and overall recall is 15.0 points above
the baseline.

## The default configuration is unchanged

With feature-hashing embeddings the app keeps the weighted sum, and every metric
is identical to the baseline: hit@5 0.800, recall@5 0.794, MRR 0.709, nDCG@10
0.736. Nobody sees a regression; the gain is available to anyone who selects a
semantic model.

## Reranking

Verified on the 15-case cross-lingual slice with `gemini-3.5-flash-lite`:

| Metric | Fusion only | Fusion + LLM rerank |
|---|---:|---:|
| hit@5 | 0.933 | **1.000** |
| recall@5 | 0.900 | **0.967** |
| MRR | 0.805 | **1.000** |
| nDCG@10 | 0.841 | **0.981** |

Reranking put the right chunk first on every case in that slice. The full
98-case reranked run could not be completed: the daily quota on the Gemini key
was exhausted partway through, so **the whole-set reranked figures are not
measured**. Reranking is off by default and costs one model call per question.

Phase 3's acceptance criterion was nDCG@10 ten points above Phase 2. Fusion
alone delivers 5.1 points. The cross-lingual slice suggests reranking covers the
rest, but that is an inference from a slice, not a measurement of the whole set.

## Confidence, recalibrated

The fused score cannot carry confidence. Rank fusion puts the best chunk at or
near 1.0 for almost every query, answerable or not. Measured on the golden set,
the top fused score had a median of 0.999 for answerable questions and 0.998 for
unanswerable ones.

Confidence is now `0.55 * relevance + 0.45 * coverage`, where relevance is the
reranker's verdict on the top chunk when a reranker ran, and otherwise how many
standard deviations the best chunk's raw similarity sits above the corpus mean.
That statistic does not depend on the embedding model's own scale, which a fixed
cosine threshold would.

## Abstention is still weak, and this is why

Abstention accuracy stays at 0.125. The measured trade-off is bad: the best
threshold found catches 88 percent of unanswerable questions but refuses 22
percent of answerable ones, which is a worse product than answering the eight.
Two honest reasons not to tune further here:

- Eight negative cases cannot support a fitted threshold. Any gain would be a
  measurement of those eight questions, not of the system.
- The signal that actually separates the two is a reranker judging whether a
  passage answers the question, and reranking is off by default.

Fixing this properly needs many more labelled negatives and a gate that reads
the rerank score. It is left open rather than tuned into a number that looks
better on this page.

## A note on the RRF constant

`RRF_K` is 60, the conventional value. A sweep over 10, 30 and 60 produced
identical metrics on this corpus: with 31 chunks the rank differences are small
either way, and the ranking is driven by which signals vote rather than by fine
rank positions. The constant would start to matter on a corpus large enough to
produce long candidate lists.

---

# Review pass — defects found re-reading Phases 0 to 3

Recorded 2026-09-06. A second read of everything above, with the eval harness
used to check each suspicion rather than argue about it.

## Citation precision had regressed under rank fusion

Fused rank scores sit near 1.0 for every returned chunk, so the citation floor
of "45 percent of the top score" stopped filtering anything: with a semantic
embedding every one of the five returned chunks was cited, relevant or not.
Measured on the golden set:

| Configuration | Chunks cited (mean) | Citation precision |
|---|---:|---:|
| Feature hashing, weighted sum | 2.93 | 0.472 |
| Semantic, rank fusion, before the fix | 5.00 | **0.222** |
| Semantic, rank fusion, after the fix | 3.42 | **0.402** |

The gate now compares raw similarities on a model-free scale: under rank
fusion, a chunk is cited when its standout (standard deviations above the corpus
mean) is within `CITATION_STANDOUT_GAP` of the best chunk's. The gap was swept:

| Gap | Cited | Precision | At least one relevant cited | Relevant kept |
|---:|---:|---:|---:|---:|
| 0.5 | 1.93 | 0.608 | 0.856 | 0.855 |
| 1.0 | 2.69 | 0.482 | 0.900 | 0.930 |
| **1.5** | 3.39 | 0.401 | 0.944 | 0.983 |
| 2.0 | 3.88 | 0.349 | 0.956 | 0.994 |
| none | 5.00 | 0.222 | 0.956 | 1.000 |

1.5 keeps 98 percent of relevant chunks and nearly doubles precision. Tighter
gaps trade relevant citations for cleaner lists, which is the wrong direction
for a system whose value is showing its evidence. When a reranker has run, its
blended score has a meaningful scale and the original floor applies.

## Rank fusion broke ties by insertion order

Tied values were numbered 1, 2, 3 in corpus order, so among chunks a signal
could not tell apart, the one inserted first scored higher. Sparse signals
produce exactly that pattern: many chunks at 1.0 on `entity` or `numeric`, the
rest at 0. Ties now share the best rank in their group. No golden-set metric
moved, which is expected: ties rarely decide the top of a list led by a dense
signal, but the ordering is now determined by the data rather than by the order
documents happened to be uploaded.

## API keys were still reaching logs and clients on three paths

Phase 3 redacted keys in the reranking path only. The answer path
(`providers.generate`), every embedding request, and three HTTP error details
returned to the browser still used httpx's own error text, which embeds the
request URL, and for Gemini that URL carries the key. Every provider response
now goes through one redacting check, and every client-facing error detail is
redacted. Redaction also covers keys in prose and bearer tokens, in case a
provider echoes them in a body.

## Smaller corrections

- A fresh database was created with the pre-migration schema and then
  immediately rebuilt into the current one. It is now created current, and the
  migration runs only for databases that actually predate it.
- `/api/system/info` could trigger an index reload on the event loop thread.
- The generation path had no retry; transient failures and rate limits now get
  the same backoff as embeddings and reranking.
- The golden set has 98 cases, not the 89 stated above; the counts are corrected.
- Dead code from before the index (`_bm25`, unused imports) is removed.

Quality after the review pass is unchanged on every metric for both
configurations. Latency at 20,000 chunks stays under 20 ms.

---

# STAIR گام ۰ — خط پایه سنجه ساختاری

ثبت‌شده در ۱۴۰۵/۰۶/۱۵ (2026-09-06). مبنای مقایسه گام‌های ۱ تا ۳ نقشه
`docs/STAIR_PLAN.fa.md` همین اعداد است، نه خط پایه فاز ۰.

بازتولید:

```
python -m eval.run_eval --json eval/results/step0.json --markdown eval/results/step0.md
```

## آنچه اضافه شد

دو سند ساختاردار به پیکره (`handbook-structured.fa.md` با سرفصل سه‌سطحی و
`bylaws-structured.fa.txt` با ساختار فصل/ماده/تبصره)، پانزده رکورد با تگ
`structural` که هرکدام `relevant_section` اعلام می‌کنند، یک رکورد نگهبان با تگ
`structural_guard` روی سندی تخت، و دو متریک `section_hit@1` و `section_recall@3`.

پیکره از ۱۳ سند و ۳۱ چانک به ۱۵ سند و ۳۵ چانک رسید. مجموعه گلدن از ۹۸ به ۱۱۴ رکورد.

## اعداد

| سنجه | فاز ۳ (۱۳ سند) | گام ۰ (۱۵ سند) |
|---|---:|---:|
| hit@5 | ۰٫۸۰۰ | ۰٫۸۰۲ |
| recall@5 | ۰٫۷۹۴ | ۰٫۷۹۷ |
| MRR | ۰٫۷۰۹ | ۰٫۷۲۸ |
| nDCG@10 | ۰٫۷۳۶ | ۰٫۷۵۰ |
| section_hit@1 | — | ۰٫۰۰۰ |
| section_recall@3 | — | ۰٫۰۰۰ |

هر دو متریک ساختاری صفرند و این درست است: تا پیش از گام ۱ هیچ چانکی
`section_path` ندارد، پس پانزده رکورد ساختاری با `blocked_by: step-1-structure`
علامت خورده‌اند. صفر بودن این دو عدد یعنی سنجه کار می‌کند، نه اینکه خراب است.

## انحراف: افت برش بین‌زبانی

| تگ | ۱۳ سند | ۱۵ سند |
|---|---:|---:|
| cross_lingual (hit@5) | ۰٫۴۰۰ | ۰٫۳۳۳ |
| fa2en (hit@5) | ۰٫۳۰۰ | ۰٫۲۰۰ |

مقایسه رکورد به رکورد نشان می‌دهد فقط چهار پرونده جابه‌جا شده‌اند و هر چهار
پرونده پیش از این هم عملاً شکست خورده بودند (MRR بین ۰٫۱۰ تا ۰٫۲۰، یعنی رتبه
پنجم تا دهم):

| پرونده | MRR پیش | MRR پس |
|---|---:|---:|
| x-fa2en-security-01 | ۰٫۲۰ | ۰٫۱۴ |
| x-fa2en-review-01 | ۰٫۱۷ | ۰٫۱۴ |
| x-fa2en-msa-03 | ۰٫۱۰ | ۰٫۰۰ |
| fa-creativity-concept-01 | ۰٫۱۴ | ۰٫۱۱ |

افت یک واحد در `fa2en` از `x-fa2en-security-01` می‌آید که از رتبه پنجم به هفتم
رفت. علتش شناخته‌شده و در همین سند ثبت شده است: امبدینگ پیش‌فرض
(هش ویژگی) شباهت بین‌زبانی صفر تولید می‌کند، پس یک سؤال فارسی درباره سندی
انگلیسی تنها با سیگنال عددی رتبه می‌گیرد و هر سند فارسی عددداری آن را پس
می‌زند. رتبه پنجم آن پرونده نتیجه شانس بود نه بازیابی.

سندهای ساختاری را عمداً عددزدایی نکردیم تا این شانس حفظ شود: آن کار نمونه
آزمون را بی‌خاصیت می‌کرد و ضعف واقعی را پنهان می‌ساخت. با امبدینگ معنایی این
برش طبق همین سند به Recall ۰٫۹۰۰ می‌رسد و مسئله موضوعیت ندارد.
