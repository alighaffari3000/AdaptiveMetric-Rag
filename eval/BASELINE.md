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

---

# STAIR گام ۱ — استخراج ساختار در ingest

ثبت‌شده در ۱۴۰۵/۰۶/۱۵. مقایسه با گام ۰ روی همان ۱۱۴ رکورد و همان ۱۵ سند.

بازتولید:

```
python -m eval.run_eval --no-cache --json eval/results/step1.json --markdown eval/results/step1.md
```

## اعداد

| سنجه | گام ۰ | گام ۱ |
|---|---:|---:|
| hit@5 | ۰٫۸۰۲ | ۰٫۸۳۰ |
| recall@5 | ۰٫۷۹۷ | ۰٫۸۳۰ |
| MRR | ۰٫۷۲۸ | ۰٫۷۵۱ |
| nDCG@10 | ۰٫۷۵۰ | ۰٫۷۷۰ |
| section_hit@1 | ۰٫۰۰۰ | ۰٫۸۶۷ |
| section_recall@3 | ۰٫۰۰۰ | ۰٫۸۶۷ |
| تأخیر p95 | ۳٫۴ms | ۳٫۲ms |

هیچ تگی پس‌رفت نکرد. تگ‌هایی که جلو رفتند:

| تگ | گام ۰ | گام ۱ |
|---|---:|---:|
| page_boundary | ۰٫۰۰۰ | ۱٫۰۰۰ |
| pdf | ۰٫۵۰۰ | ۱٫۰۰۰ |
| conceptual | ۰٫۵۷۱ | ۰٫۷۱۴ |
| multi_intent | ۰٫۷۷۸ | ۰٫۸۸۹ |
| fa2en | ۰٫۲۰۰ | ۰٫۳۰۰ |
| causal | ۰٫۷۱۴ | ۰٫۷۸۶ |
| cross_lingual | ۰٫۳۳۳ | ۰٫۴۰۰ |

‏`page_boundary` از فاز ۰ تا امروز صفر بود و `IMPROVEMENT_PLAN.fa.md` صراحتاً آن را
نمونه‌ای معرفی کرده بود که قطعه‌بندی ساختاری باید حلش کند. جمله‌ای که وسط مرز
صفحه شکسته می‌شد اکنون در یک بخش پیوسته می‌ماند. افت بین‌زبانی گام ۰ هم جبران شد.

## بن‌بستی که سر راه بود: ریز شدن چانک‌ها

نسخه اول این گام هر بخش را یک چانک کرد. نتیجه فاجعه بود:

| سنجه | گام ۰ | یک چانک به ازای هر بخش |
|---|---:|---:|
| تعداد چانک | ۳۵ | ۱۱۱ |
| MRR | ۰٫۷۲۸ | ۰٫۶۵۸ |
| nDCG@10 | ۰٫۷۵۰ | ۰٫۶۸۳ |
| temporal | ۱٫۰۰۰ | ۰٫۷۶۹ |
| follow_up | ۰٫۵۰۰ | ۰٫۲۵۰ |

برای اینکه بدانیم مقصر «ساختار» است یا «ریزدانگی»، دو پرچم به `run_eval` اضافه
شد (`--flat-chunking` و `--chunk-size`/`--chunk-overlap`) و روی همان ۹۹ رکورد
غیرساختاری، قطعه‌بندی تخت با ریزدانگی برابر سنجیده شد:

| قطعه‌بندی | چانک | hit@5 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| تخت ۹۰۰/۱۴۰ (پیش‌فرض قدیم) | ۳۵ | ۰٫۷۹۱ | ۰٫۷۲۳ | ۰٫۷۴۲ |
| تخت ۲۵۰/۳۰ | ۱۲۳ | ۰٫۷۶۹ | ۰٫۵۷۳ | ۰٫۶۱۲ |
| تخت ۲۳۰/۲۵ | ۱۳۲ | ۰٫۷۶۹ | ۰٫۵۹۶ | ۰٫۶۱۴ |
| ساختاری، یک بخش یک چانک | ۱۱۱ | ۰٫۷۴۷ | ۰٫۶۴۲ | ۰٫۶۶۸ |
| **ساختاری + بسته‌بندی ۹۰۰/۱۴۰** | **۳۵** | **۰٫۸۲۴** | **۰٫۷۳۲** | **۰٫۷۵۴** |

نتیجه صریح است: در ریزدانگی برابر (حدود ۱۲۰ چانک) ساختار از تخت بهتر است
(MRR ۰٫۶۴۲ در برابر ۰٫۵۷۳ و ۰٫۵۹۶). پس افت نسخه اول از ریزدانگی می‌آمد نه از
ساختار. با بسته‌بندی بخش‌های کوچک تا سقف `chunk_size`، تعداد چانک به همان ۳۵
برگشت و از قطعه‌بندی تخت هم جلو زد.

بسته‌بندی حقیقت را پنهان نمی‌کند: چانکی که چند بخش را در خود دارد، هر بخش را
در `section_path` اعلام می‌کند (`فصل اول > دورکاری | حضور در دفتر | ماموریت`).
مرز بالادست همچنان مرز سند است، نه پنجره‌ای کور: هیچ چانکی از وسط یک بخش شروع
یا تمام نمی‌شود مگر بخشی که خودش از `chunk_size` بلندتر باشد.

---

# STAIR گام ۲ — ساختار به‌عنوان سیگنال بازیابی

ثبت‌شده در ۱۴۰۵/۰۶/۱۵. مقایسه با گام ۱.

## اعداد

| سنجه | گام ۰ | گام ۱ | گام ۲ |
|---|---:|---:|---:|
| hit@5 | ۰٫۸۰۲ | ۰٫۸۳۰ | ۰٫۸۴۰ |
| recall@5 | ۰٫۷۹۷ | ۰٫۸۳۰ | ۰٫۸۴۰ |
| MRR | ۰٫۷۲۸ | ۰٫۷۵۱ | ۰٫۷۵۳ |
| nDCG@10 | ۰٫۷۵۰ | ۰٫۷۷۰ | ۰٫۷۷۳ |
| section_hit@1 | ۰٫۰۰۰ | ۰٫۸۶۷ | ۰٫۸۶۷ |
| section_recall@3 | ۰٫۰۰۰ | ۰٫۸۶۷ | ۰٫۸۶۷ |

| تگ | گام ۱ | گام ۲ |
|---|---:|---:|
| structural | ۰٫۸۶۷ | ۰٫۹۳۳ |
| numeric | ۰٫۷۳۸ | ۰٫۷۶۲ |
| fa2fa | ۰٫۸۴۸ | ۰٫۸۶۴ |

هیچ تگی پس‌رفت نکرد؛ بدترین جابه‌جایی صفر است.

‏`section_hit@1` تکان نخورد. دلیلش این است که پس از گام ۱، سیزده مورد از پانزده
پرونده ساختاری بخش درست را همان رتبه اول می‌آورند و دو مورد باقی‌مانده به
امبدینگ محدودند نه به ساختار. معیار پذیرش نقشه (ده واحد بالاتر از گام ۰) با
مجموع گام ۱ و ۲ محقق شده است، اما سهم گام ۲ در این متریک صفر بوده و اینجا ثبت
می‌شود.

## دو انحراف که در مسیر اصلاح شدند

**یک: سیگنال ساختار از سهم `dense` تأمین نشود.** نسخه اول وزن ساختار را از
سیگنال معنایی برداشت (مثلاً conceptual از ۰٫۵۲ به ۰٫۴۶). نتیجه: `conceptual` از
۰٫۷۱۴ به ۰٫۵۷۱ افتاد. وزن‌ها بازنویسی شدند تا `dense` هر نیت دقیقاً همان مقدار
تنظیم‌شده فاز ۲ بماند و سهم ساختار از `metadata` برداشته شود. این از نظر معنا هم
درست‌تر است: `structure` همان سرفصل‌هایی را می‌خواند که `metadata` می‌خواند، بدون
اینکه نام و نوع فایل رقیقش کنند.

**دو: واژه‌ای که به خودِ سند اشاره می‌کند شاهد نیست.** حتی با `dense` دست‌نخورده،
`conceptual` همچنان چهارده واحد افت داشت و مقصر یک پرونده بود:
«موضوع این کتاب چیه؟». واژه «موضوع» در سرفصل «ماده ۲ — موضوع قرارداد» هست، پس
سیگنال ساختار دو قرارداد را بالای کتابی که سؤال درباره‌اش بود نشاند. مجموعه
`DOCUMENT_WORDS` در `app/retrieval.py` این واژه‌ها را از سیگنال ساختار کنار
می‌گذارد (موضوع، خلاصه، درباره، کتاب، متن، topic، subject، summary و مانند آن‌ها).
این واژه‌ها در بقیه سیگنال‌ها دست‌نخورده می‌مانند؛ متنی که پر از آن‌هاست شاهد
واقعی است، سرفصلی که تصادفاً یکی‌شان را دارد نیست.

---

# STAIR گام ۳ — مسیریاب فهرست مطالب (پیش‌فرض خاموش)

ثبت‌شده در ۱۴۰۵/۰۶/۱۵.

## با فلگ خاموش: تطابق کامل

```
python -m eval.run_eval --no-cache --json eval/results/step3.json
```

خلاصه و همه تگ‌ها با گام ۲ یکسان‌اند و هر ۱۱۴ پرونده رکورد به رکورد برابرند
(به‌جز تأخیر که نویز اندازه‌گیری است). معیار پذیرش «تطابق بیت‌به‌بیت با فلگ
خاموش» محقق شد.

## با فلگ روشن: اندازه‌گیری روی گلدن‌ست انجام نشد

مسیریاب به یک provider تولید پاسخ نیاز دارد و در این محیط کلید API در دسترس
نیست. رفتارش با provider ساختگی در `tests/test_toc_router.py` پوشش داده شده
(۳۳ آزمون)، اما عدد Recall روی گلدن‌ست با مسیریاب روشن اندازه‌گیری **نشده**
است. سربار تأخیر هم به همین دلیل اندازه‌گیری نشده. فلگ به همین سبب پیش‌فرض
خاموش می‌ماند.

## دو نقصی که هنگام نوشتن آزمون‌ها پیدا شد

**یک: باریک کردن دامنه، اطمینان را نابود می‌کرد.** `standout` به‌صورت فاصله
معیار بهترین چانک از میانگین همان مجموعه‌ای حساب می‌شد که جستجو رویش انجام
می‌شد. وقتی مسیریاب دامنه را به یک بخش می‌رساند، آن بخش طبق تعریف نسبت به
خودش برجسته نبود، `standout` صفر می‌شد و دروازه شواهد پاسخ درست را رد می‌کرد.
حالا `standout` روی دامنه‌ای حساب می‌شود که کاربر جستجو می‌کند، پیش از هر
باریک‌سازی مسیریاب. با فلگ خاموش این تغییر بی‌اثر است و اعداد بالا همین را
نشان می‌دهند.

**دو: جداکننده مسیر مانع تطبیق می‌شد.** مدلی که
`فصل دوم - مرخصي استعلاجي` می‌نویسد همان بخشی را می‌گوید که در فهرست
`فصل دوم — مرخصی > مرخصی استعلاجی` نوشته شده. نرمال‌سازی حالا `>` را هم مثل
بقیه نقطه‌گذاری به فاصله تبدیل می‌کند. شباهت با مسیر درست ۰٫۸۸ و با بخش خواهرش
(`مرخصی استحقاقی`) ۰٫۷۶ است، پس آستانه ۰٫۸۶ آن دو را قاطی نمی‌کند.

---

# STAIR گام ۴ — نمونه منفی و اندازه‌گیری دوباره امتناع

ثبت‌شده در ۱۴۰۵/۰۶/۱۵.

## آنچه اضافه شد

سی پرسش بی‌پاسخ بازبینی‌شده با تگ `near_miss`، که مجموع نمونه‌های منفی را از ۸ به
۳۸ رساند. هر کدام عمداً «نزدیک» است: موضوعی که کتابخانه پوشش می‌دهد ولی همان
واقعیت مشخص را ندارد (مثلاً «مرخصی زایمان چند روز است؟» وقتی آیین‌نامه فقط
مرخصی استحقاقی و استعلاجی دارد). پرسش نامرتبط با کل کتابخانه فقط به‌خاطر عدم
تطابق واژگانی رد می‌شود و چیزی را نمی‌سنجد.

اسکریپت `eval/make_questions.py` هم اضافه شد که به سبک SearchTome از روی هر
چانک پرسش پیش‌نویس می‌کند. خروجی‌اش در `eval/golden.generated.jsonl` می‌نشیند،
**نه** در `golden.jsonl`، و آزمون
`test_no_generated_record_has_leaked_into_the_golden_set` این مرز را نگه می‌دارد.
در این محیط کلید API نبود، پس سی نمونه بالا دستی نوشته و بازبینی شدند.

## نتیجه: امتناع تقریباً کار نمی‌کند، و حالا این را می‌دانیم

| سنجه | پیش از گام ۴ | پس از گام ۴ |
|---|---:|---:|
| تعداد نمونه منفی | ۸ | ۳۸ |
| دقت امتناع | ۰٫۱۲۵ (۱ از ۸) | ۰٫۰۷۹ (۳ از ۳۸) |
| دقت امتناع روی همان ۸ نمونه قدیمی | ۰٫۱۲۵ (۱ از ۸) | ۰٫۳۷۵ (۳ از ۸) |
| answerable_not_abstained | ۰٫۹۶۲ | ۰٫۹۷۲ |

عدد ۰٫۱۲۵ قبلی یک از هشت بود و معنایی نداشت. عدد واقعی حدود هشت درصد است.

## نقصی که پیدا شد: امتیاز نسبی به‌جای مطلق در دروازه شواهد

شرط `top_features["bm25"] >= .08` در `select_grounded` روی امتیازی کار می‌کرد که
به بهترین کاندیدا نرمال شده است (`lexical[position] / bm_max`). یعنی چانک اول
هر وقت **هر** کاندیدایی امتیاز واژگانی غیرصفر داشت، عدد ۱٫۰ می‌گرفت. اندازه‌گیری:
در ۸۱ درصد پرسش‌های گلدن‌ست چانک اول دقیقاً ۱٫۰ بود. یعنی این شرط در عمل
می‌پرسید «آیا بازیابی چیزی برگرداند؟».

جایگزین `_query_coverage` است: سهم واژه‌های متمایز سؤال که در متن چانک هستند.
مقیاسش به بقیه کاندیداها وابسته نیست، که همان چیزی است که یک دروازه لازم دارد.
اثرش: هیچ سنجه بازیابی تکان نخورد، `answerable_not_abstained` نه‌دهم واحد بهتر
شد، و امتناع روی هشت نمونه قدیمی از یک به سه رسید.

## چرا آستانه را دوباره تنظیم نکردیم

با ۳۸ نمونه منفی حالا می‌شود پرسید آیا اصلاً آستانه‌ای هست که جدا کند:

| سیگنال | AUC |
|---|---:|
| اطمینان فعلی | ۰٫۶۱۶ |
| پوشش واژه‌های سؤال در چانک اول | ۰٫۷۲۷ |
| شباهت متراکم خام | ۰٫۵۷۴ |

‏(۰٫۵ یعنی هیچ اطلاعاتی ندارد.) بهترین سیگنال، یعنی پوشش واژگانی، این مرز را
می‌سازد:

| حداقل پوشش | دقت امتناع | نگهداری پرسش‌های پاسخ‌پذیر |
|---:|---:|---:|
| ۰٫۲۰ | ۰٫۱۰۵ | ۰٫۹۲۵ |
| ۰٫۳۵ | ۰٫۴۲۱ | ۰٫۷۹۲ |
| ۰٫۴۵ | ۰٫۶۳۲ | ۰٫۷۳۶ |
| ۰٫۶۰ | ۰٫۸۱۶ | ۰٫۶۳۲ |

نرخ مبادله تقریباً «یک پاسخ درست از دست رفته به ازای هر پرسش بی‌پاسخ گرفته‌شده»
است. برای دستیاری که پاسخش را با ارجاع نشان می‌دهد، امتناع اشتباه از پاسخ
محتاطانه بدتر است. آستانه روی ۰٫۲۰ ماند و دلیلش همین جدول است، نه حدس.

این تحلیل با امبدینگ پیش‌فرض (هش ویژگی) انجام شده. سیگنالی که جدا کند احتمالاً
از امبدینگ معنایی می‌آید، نه از تنظیم آستانه؛ همان نتیجه‌ای که فاز ۳ هم به آن
رسیده بود، ولی حالا با ۳۸ نمونه به‌جای ۸ نمونه.

## نقص سوم: همان توکن معیوب، این بار در محاسبه اطمینان

آزمون سرتاسری روی یک کتابخانه تک‌سندی نشان داد سؤالی که سند صریحاً پاسخش را
دارد رد می‌شود. علت همان «است؟» بود: `score_confidence` پوشش را روی
`query_tokens` خام حساب می‌کرد، پس مخرج یک واحد بزرگ‌تر می‌شد و اطمینان از
۰٫۲۷ به ۰٫۲۲۵ می‌افتاد، درست زیر آستانه ۰٫۲۴.

حالا هم دروازه و هم اطمینان از یک مجموعه واژه استفاده می‌کنند
(`_query_coverage`). روی گلدن‌ست هیچ سنجه‌ای تکان نخورد و
`answerable_not_abstained` همان ۰٫۹۷۲ ماند؛ اثرش روی کتابخانه‌های کوچک است که
گلدن‌ست نمونه‌ای از آن‌ها ندارد.

## نقص دوم که پیدا شد و عمداً کامل اصلاح نشد

نقطه‌گذاری فارسی داخل بلوک عربی یونیکد است، پس بازه `؀-ۿ` در
`TOKEN_RE` آن را می‌بلعد: «است؟» یک توکن واحد می‌شود، هرگز با فهرست ایست‌واژه
تطبیق نمی‌خورد، و همه‌جا از BM25 تا دروازه شواهد به‌عنوان یک واژه سؤال شمرده
می‌شود.

اصلاح کامل (پاک کردن این نشانه‌ها داخل `tokenize`) اندازه‌گیری شد و **۱٫۹ واحد
از hit@5 کم کرد**؛ `conceptual` چهارده واحد و `pdf` هفده واحد. علتش دو پرونده
است که هیچ واژه مشترکی با سند هدفشان ندارند:

| پرونده | واژه‌های سؤال موجود در سند هدف |
|---|---|
| fa-creativity-concept-01 («موضوع این کتاب چیه؟») | هیچ‌کدام |
| x-fa2en-review-01 (سؤال فارسی، PDF انگلیسی) | هیچ‌کدام |

هر دو با هش ویژگی شانسی درست رتبه می‌گرفتند و با تغییر توکن‌ها شانس جابه‌جا شد.

پس نقطه‌گذاری فقط جایی پاک می‌شود که دروازه واژه‌ها را با سؤال می‌سنجد
(`_query_coverage`)، و `tokenize` که ایندکس BM25 و بردارها را می‌سازد دست‌نخورده
ماند. با این کار هیچ تگی تکان نخورد. اصلاح کامل وقتی معنا پیدا می‌کند که
امبدینگ پیش‌فرض معنایی باشد و این دو پرونده با سیگنال رتبه بگیرند نه با شانس.

---

# STAIR — بازبینی کد و رفع ایرادها

ثبت‌شده در ۱۴۰۵/۰۶/۱۶ (2026-09-07). بازبینی کامل تغییرات گام ۰ تا ۴، ده ایراد
پیدا شد و همه بازتولید و رفع شدند. اعداد بازیابی تکان نخوردند:

| سنجه | پس از گام ۴ | پس از بازبینی |
|---|---:|---:|
| hit@5 | ۰٫۸۴۰ | ۰٫۸۴۰ |
| MRR | ۰٫۷۵۳ | ۰٫۷۵۳ |
| nDCG@10 | ۰٫۷۷۳ | ۰٫۷۷۳ |
| section_hit@1 | ۰٫۸۶۷ | ۰٫۸۶۷ |
| answerable_not_abstained | ۰٫۹۷۲ | ۰٫۹۶۲ |
| آزمون‌ها | ۲۲۱ | ۲۴۳ |

## ریشه مشترک سه ایراد: رشته به‌جای داده

مسیر بخش یک چانک بسته‌بندی‌شده به شکل `A > B > x | y` نوشته می‌شد و بعد قرار بود
دوباره تجزیه شود. این تجزیه ممکن نیست: وقتی یک دنباله خودش جداکننده دارد
(`A > B | C > d`) معلوم نیست دو بخش بوده یا سه. نتیجه‌اش سه ایراد بود:

- فهرست مطالب `handbook-structured.fa.md` به‌جای ۱۶ مدخل، **دو سطر بی‌معنا** بود.
  همین فهرست ورودی مسیریاب گام ۳ است.
- `matching_positions` روی همان رشته کار می‌کرد.
- `build_toc` هم.

حالا فهرست بخش‌ها به‌صورت `sections` (یک لیست واقعی) در `chunks.metadata` ذخیره
و از `Snapshot.rows` خوانده می‌شود. `section_path` فقط برچسب نمایشی است و
داک‌استرینگش صریحاً می‌گوید که قابل تجزیه نیست. ردیف‌های قدیمی به `section_path`
برمی‌گردند.

## مسیریاب روی کتابخانه چندسندی همیشه بی‌اثر بود

‏`toc_from_rows` مسیرها را با نام سند پیشوند می‌زد ولی `matching_positions` با
`section_path` بدون پیشوند مقایسه می‌کرد، پس هیچ‌وقت چیزی مطابقت نمی‌کرد: یک
فراخوانی مدل و تا ۲۰ ثانیه تأخیر، بدون هیچ اثری. حالا هر مدخل هم `label`
(چیزی که به مدل نشان داده می‌شود) دارد و هم `path` (چیزی که چانک حمل می‌کند)، و
`constrain` همیشه `path` را برمی‌گرداند.

## اطمینان، چانکی را توصیف می‌کرد که برنگشته بود

‏`standout` بیشینه پیکره را گزارش می‌کرد. یعنی مسیریابی که به بخش اشتباه می‌رفت،
باز هم اطمینان بخش درست را نشان می‌داد. حالا بیشینه‌ی چانک‌های **بازگشتی** است:
مقیاس از پیکره می‌آید (پس باریک‌سازی بخش را نسبت به خودش بی‌اهمیت نمی‌کند) ولی
مقدار از چیزی که کاربر می‌بیند. هزینه‌اش یک پرونده در
`answerable_not_abstained` بود (۰٫۹۷۲ به ۰٫۹۶۲).

همچنین `standout` منفی دیگر از اطمینان کم نمی‌کند؛ چانک زیر میانگین «متمایز
نیست»، نه «شاهد مخالف».

## موقعیت ردیف در برابر شناسه چانک

مسیریاب موقعیت ردیف‌ها را از یک snapshot می‌خواند و `retrieve` روی snapshot بعدی
اعمالشان می‌کرد. یک حذف در این فاصله، جستجو را بی‌صدا به چانک‌های دیگری می‌برد.
حالا شناسه چانک منتقل می‌شود که پایدار است.

## مرزبندی بسته‌بندی: داک‌استرینگ اصلاح شد، نه کد

داک‌استرینگ `chunk_blocks` قول می‌داد بسته‌بندی از یک بخش سطح‌بالا عبور نمی‌کند،
ولی چنین بررسی‌ای وجود نداشت و `top_level` کد مرده بود. مرزبندی پیاده و
اندازه‌گیری شد:

| حالت | چانک | hit@5 | MRR | page_boundary |
|---|---:|---:|---:|---:|
| بدون مرز (فعلی) | ۳۵ | ۰٫۸۴۰ | ۰٫۷۵۳ | ۱٫۰۰۰ |
| با مرز در عمق انشعاب | ۸۸ | ۰٫۷۸۳ | ۰٫۶۸۰ | ۰٫۰۰۰ |

پنج و هفت‌دهم واحد hit@5 و برگشت page_boundary به صفر. مرز حذف شد، کد مرده پاک
شد و داک‌استرینگ حالا می‌گوید `size` تنها مرز است و چرا: بسته‌بندی فقط بخش‌های
کوچک‌تر از `size` را ادغام می‌کند، پس در سندی که فصل‌هایش به‌قدر کافی بلندند،
هر فصل به‌هرحال جدا بریده می‌شود.

## استخراج ساختار

- متن سرفصل در HTML و DOCX از محتوای چانک حذف می‌شد (در Markdown و PDF نمی‌شد).
  یعنی واژه‌ای که فقط در سرفصل بود، در ایندکس BM25 وجود نداشت.
- `---` پایانی front matter یک سرفصل setext تلقی می‌شد و بالای سند زیر عنوانی
  ساختگی مثل `author: Ops` بایگانی می‌شد.
- «بند اول این است که همه کارکنان باید حضور داشته باشند» سرفصل تشخیص داده
  می‌شد. حالا بیش از ۹ کلمه یا پایان جمله‌ای، متن است نه عنوان.

## پاسخ استخراجی محلی سؤال را نمی‌خواند

‏`local_answer` اولین جمله بلندتر از ۳۵ نویسه را برمی‌داشت، بدون نگاه به سؤال:
دو سؤال متفاوت از یک چانک، یک جواب می‌گرفتند. تا وقتی چانک‌ها پنجره‌های کوچک
بودند و هر چانک تقریباً یک واقعیت داشت پنهان ماند؛ حالا که چانک یک بخش کامل
(یا چند بخش) است، انتخاب جمله اجباری است.

‏`best_evidence` بازنویسی شد تا چانک را در مرز سرفصل‌ها به گروه (سرفصل، جملات)
بشکند، گروه را با تطابق سرفصل **و** بدنه امتیاز دهد، و از بدنه گروه برنده نقل
کند. این همان تقسیم کاری است که مقاله توصیف می‌کند: سرفصل بخش را پیدا می‌کند،
متن زیر آن پاسخ می‌دهد. سه سؤال روی یک چانک سه‌بخشی، حالا هر سه از بخش درست
پاسخ می‌گیرند؛ پیش از این هر سه یک جواب می‌گرفتند.

## ایراد کوچک

شناسه‌های `eval/make_questions.py` با الگوی `index * 10 + offset` ساخته می‌شدند
و با `--per-chunk` بیشتر از ۱۰ برخورد می‌کردند. حالا یک شمارنده واحد است و
برخورد احتمالی گزارش می‌شود، نه اینکه بی‌صدا حذف شود.


## بازبینی دوم: هشت ایراد در خودِ اصلاحات

بازبینی دوباره روی اصلاحات بالا هشت ایراد دیگر پیدا کرد که بیشترشان در کدی
بودند که همان دور نوشته شده بود. اعداد بازیابی باز هم تکان نخوردند.

**تشخیص سرفصل بیش از حد سخت‌گیر شده بود.** «.» و «:» به فهرست پایان‌جمله اضافه
شده بود، پس `تبصره:`، `ماده ۱۲:`، `Section 4.` و `Article 5:` که همه پیش از آن
سرفصل شناخته می‌شدند، رد می‌شدند. حالا فقط شمارش کلمه (بیش از ۹) و ویرگول و
علامت سؤال متن را از عنوان جدا می‌کند؛ `_clean_title` هم دونقطه پایانی را
از قبل حذف می‌کرد، که نشان می‌داد این شکل‌ها قرار بوده پشتیبانی شوند.

**هر `---` ابتدای فایل front matter فرض می‌شد.** یک خط جداکننده افقی در ابتدای
سند باعث می‌شد همه سرفصل‌ها تا `---` بعدی بی‌صدا حذف شوند. حالا وجود دست‌کم یک
کلید YAML شرط است.

**عدد نقطه‌دار در متن، جمله را غیرقابل نقل می‌کرد.** `7.4.0 fixes the memory leak`
از الگوی `4.2.1 Rollback procedure` قابل تشخیص نیست. هنگام ingest یک تشخیص
اشتباه فقط یک مرز اضافه می‌سازد، ولی هنگام تقسیم چانک برای نقل‌قول، همان خط را
از دسترس خارج می‌کند. `looks_like_heading` حالا پارامتر `unambiguous` دارد و
تقسیم چانک از قاعده عدد نقطه‌دار استفاده نمی‌کند.

**سرفصل بر بدنه می‌چربید.** امتیاز گروه `۰٫۶۰ × سرفصل + ۰٫۴۰ × امتیاز جمله` بود،
ولی امتیاز جمله با `answer` خالی حداکثر حدود ۰٫۴۵ می‌شود. نتیجه: سرفصلی که یک‌سوم
واژه‌های سؤال را داشت، بخشی را می‌برد که بدنه‌اش همه‌شان را داشت. حالا هر دو با
یک سنجه اندازه‌گیری می‌شوند: `۰٫۴۵ × سرفصل + ۰٫۵۵ × بدنه`.

**گروهی که فقط سرفصل است نباید برنده شود.** یک عنوان فصل که بلافاصله زیرعنوان
دارد، بدنه ندارد و چیزی برای پاسخ ندارد. حالا فقط وقتی انتخاب می‌شود که کل چانک
چیزی جز سرفصل نداشته باشد.

**پیشوند نام سند در مسیریاب بی‌اثر بود.** `constrain` مسیر بدون پیشوند را
برمی‌گرداند، پس دو سندی که بخش هم‌نام دارند هر دو انتخاب می‌شدند. حالا کل مدخل
برگردانده می‌شود و `matching_positions` نام سند را هم بررسی می‌کند.

**ردیف‌های قدیمی با برچسب ادغام‌شده.** ردیفی که پیش از این تغییر نوشته شده،
`Root > A | B | C` را به‌عنوان یک بخش واحد گزارش می‌کرد، پس مسیریاب
`Root > B` را پیدا نمی‌کرد و می‌توانست جستجو را به مجموعه‌ای ببرد که چانک حاوی
پاسخ در آن نیست. حالا در همین مسیر پشتیبان، برچسب به‌صورت best-effort تجزیه
می‌شود.

**سرفصل Word که تنها می‌ماند.** سرفصلی که پاراگراف بعدی‌اش از `chunk_size`
بزرگ‌تر بود، خودش یک چانک می‌شد: چانکی بدون محتوا که در بازیابی رقابت می‌کند و
بدنه بخش هم واژه سرفصل را ندارد. حالا سرفصل به اولین پاراگراف بخشش می‌چسبد.


## بازبینی سوم: شش ایراد، همه ناشی از اصلاح بیش از حد

**تشخیص سرفصل باید به دنباله نگاه کند، نه به پایان خط.** حذف «.» از فهرست
پایان‌جمله همان ایرادی را برگرداند که دور قبل رفع شده بود:
`Section 3 covers rollback and recovery.` دوباره سرفصل می‌شد. شمارش کلمه هم
نجاتش نمی‌داد چون این جمله‌ها پنج تا هشت کلمه‌اند.

تفاوت واقعی جای دیگری است: `Section 4.` و
`Section 3 covers rollback and recovery.` هر دو با نقطه تمام می‌شوند، ولی در
اولی بعد از شماره چیزی نیست و در دومی جمله ادامه پیدا می‌کند. الگوها حالا فقط
شماره‌گذاری را می‌گیرند و `_titleish_tail` دنباله را می‌سنجد: خالی، یا شروع با
جداکننده (`—`، `:`، `.`)، یا برچسبی کوتاه بدون نقطه پایانی. با این قاعده هر
دوازده شکل واقعی سرفصل پذیرفته و هر پنج نمونه متن رد می‌شوند.

**کلید front matter همیشه انگلیسی نیست.** الگوی کلید فقط حروف ASCII را
می‌پذیرفت و اولین خط نامنطبق کل تشخیص را لغو می‌کرد، پس در یک پیکره فارسی‌محور
front matter شناسایی نمی‌شد و `---` پایانی‌اش دوباره سرفصل setext می‌شد.

**گروه فقط-سرفصل نباید حذف شود، باید رتبه بگیرد.** بندی که کل حکم را روی خط
شماره‌دار می‌نویسد (`ماده ۵: سقف مرخصی استحقاقی سی روز است`) بدنه ندارد؛ حذف
کامل چنین گروه‌هایی آن را غیرقابل نقل می‌کرد. حالا امتیاز گروه یک زوج است و
داشتن بدنه فقط تساوی را می‌شکند.

**سرفصل‌های پشت سر هم در Word.** اصلاح قبلی سرفصل معلق را وقتی سرفصل بعدی
می‌رسید تخلیه می‌کرد، پس «عنوان سند + عنوان فصل + پاراگراف بلند» باز هم یک چانک
فقط-سرفصل می‌ساخت. آزمون قبلی این را نمی‌گرفت چون `chunk_size` را ۹۰۰ گذاشته
بود. حالا سرفصل‌های متوالی با هم منتظر اولین پاراگراف می‌مانند.

**بخش‌های هم‌نام در دو سند در هم می‌رفتند.** `setdefault` فقط اولی را نگه
می‌داشت، پس پاسخ کوتاه مدل (فقط عنوان بخش) جستجو را به یک سند دلخواه محدود
می‌کرد و دیگری غیرقابل دسترس می‌شد. حالا هر مدخلی که پاسخ می‌تواند به آن اشاره
کند نگه داشته می‌شود.

**سرفصل HTML و Word هیچ نشانه‌ای در متن ندارد.** بازسازی سرفصل‌ها از روی متن
چانک، `Rollback procedure` را جمله می‌خواند و در پاسخ به «what is the rollback
procedure?» خودِ عنوان را نقل می‌کرد. حالا فهرست `sections` خودِ چانک مرجع است
و `best_evidence` آن را می‌گیرد؛ بازسازی از متن فقط پشتیبان است.

بررسی سرتاسری شش پرسش روی سه قالب سند (Markdown با front matter، متن حقوقی
فارسی، HTML) هر شش را از بخش درست پاسخ می‌دهد.


## بازبینی چهارم: علت ریشه‌ای، نه وصله دیگر

سه دور اصلاح، هر بار ایراد تازه‌ای ساخت. علتش دو چیز ساختاری بود:

**قاعده سرفصل در دو جا نوشته شده بود.** `looks_like_heading` به‌جای فراخوانی
`_pattern_level`، خودش الگوها را دوباره امتحان می‌کرد و در نتیجه از دروازه
`_titleish_tail` رد می‌شد. یعنی خطی که `detect_plain` متن می‌دانست، هنگام
تقسیم چانک برای نقل‌قول همچنان سرفصل شمرده می‌شد و جمله پاسخ‌دهنده غیرقابل نقل
می‌ماند — دقیقاً همان ایرادی که دور قبل ادعای رفعش را داشت. حالا یک پیاده‌سازی
هست و آزمون هر دو مسیر ورودی را با هم می‌سنجد.

**سقف چهار کلمه‌ای دنباله، اختیاری و غلط بود.** سرفصل‌های واقعی مثل
`Chapter 3 Network security and incident response` یا
`فصل سوم شرایط عمومی استخدام کارکنان دولت` را رد می‌کرد. تفاوت واقعی میان
این‌ها و `Section 3 covers rollback and recovery.` طول نیست، نقطه پایانی است.
سقف حذف شد و شمارش کلمه کل خط (که از قبل بود) تنها قاعده طول ماند.

**جای درست رفع «چانک سرگردان»، `chunk_blocks` است نه خواننده DOCX.** اصلاح
قبلی سرفصل‌ها را در خواننده Word به هم می‌چسباند، که مسیر بخش اول را از دست
می‌داد (`Alpha` زیر نام `Beta` بایگانی می‌شد). حالا استخراج صادقانه است — هر
سرفصل بلوک خودش با مسیر خودش — و `chunk_blocks` هر بلوک کوچکِ در انتظار را به
اولین پنجره بخش بزرگ بعدی می‌چسباند. این برای همه قالب‌ها کار می‌کند، نه فقط
Word، و هیچ بخشی از فهرست حذف نمی‌شود.

سه ایراد کوچک‌تر هم رفع شد: یک پاسخ مبهم مسیریاب (عنوانی که در چهار سند هست)
کل سهمیه را می‌خورد و پیشنهاد دقیق بعدی را دور می‌ریخت، پس نتایج حالا
گردشی برداشته می‌شوند؛ مسیر بازنویسی ارجاع‌های ذخیره‌شده در `main.py` فهرست
بخش‌ها را دریافت نمی‌کرد (و کوئری‌اش هم آن را نمی‌خواند)؛ و وقتی هیچ جمله‌ای با
سؤال مشترکی ندارد، تساوی با کوتاه‌ترین شکسته می‌شد و برچسبی بی‌محتوا برمی‌گشت،
که حالا بلندترین خط است.

خواندن فهرست بخش‌های یک چانک از پایگاه‌داده هم به `structure.sections_of` منتقل
شد تا `index` و `main` یک پیاده‌سازی داشته باشند.


## بازبینی پنجم: مرز ابهام و مسیر سرفصل

**سقف طول دنباله برگشت، این بار با دلیل ثبت‌شده.** حذفش در دور قبل باعث شد
جمله‌های پنج تا نه کلمه‌ای سرفصل شوند
(`Section 5 lists the approved vendors and contacts`)، یعنی مسیر بخش ساختگی
ساخته شود. هیچ طولی این دو را جدا نمی‌کند —
`Chapter 3 Network security and incident response` و
`Chapter 4 was written by the finance team` دنباله‌های پنج و شش کلمه‌ای دارند —
پس برش عمداً روی سمت محتاطانه ابهام گذاشته شد: قاعده این ماژول این است که مسیر
غلط از نبودن مسیر بدتر است. سندی که سرفصل‌هایش را با خط تیره یا دونقطه جدا
می‌کند، در هر طولی سالم می‌ماند.

**شمارش کلمه از قاعده کلیدواژه جدا شد.** پیش از این سقف نه کلمه روی کل خط بود و
سرفصل‌های بلندِ نقطه‌گذاری‌شده را هم رد می‌کرد
(`فصل سوم — شرایط عمومی استخدام کارکنان دولت و نهادهای وابسته`). حالا شمارش
کلمه فقط نگهبان قاعده عدد نقطه‌دار است، که نقطه‌گذاری‌ای برای خواندن ندارد.

**سرفصل هرگز چانک قبلی را نمی‌بندد.** اصلاح دور قبل فقط بلوک بزرگ‌تر از `size`
را پوشش می‌داد، پس سرفصل در حالت عادی به انتهای چانک بخش قبلی می‌چسبید و متن
خودش بدون آن در چانک بعدی می‌ماند — دقیقاً واژه‌ای که خواننده جستجو می‌کند.
`chunk_blocks` حالا هر دنباله سرفصلِ بی‌متن را از انتهای چانک جدا می‌کند و به
بخشی که باز می‌کند می‌برد، و سندی که با سرفصل تمام می‌شود آن را به چانک آخر
می‌چسباند به‌جای اینکه چانکی بدون متن بسازد.

**صفحه و بخش در برش بخش‌های بلند درست شدند.** پنجره‌ای که تماماً از متن حمل‌شده
ساخته شده بود، بخش بلند را هم اعلام می‌کرد و با شماره صفحه آن مهر می‌خورد؛ یعنی
متن صفحه یک به‌عنوان صفحه دو ارجاع داده می‌شد. حالا فقط پنجره‌ای که واقعاً
سرفصل‌های بازکننده را دارد آن‌ها را اعلام می‌کند، و شماره صفحه از خودِ آن
بلوک‌ها می‌آید.


## بازبینی ششم: سوراخ جداکننده و کران‌های بسته‌بندی

**جداکننده، متن را سرفصل می‌کرد.** با حذف شمارش کلمه در دور قبل، هر جمله‌ای که
با کلیدواژه و یک جداکننده شروع می‌شد سرفصل می‌شد، چون `_titleish_tail` روی
جداکننده کوتاه‌مدار می‌شد و به آزمون نقطه پایانی نمی‌رسید:
`Section 4. This section describes the rollback procedure.` سرفصل شمرده می‌شد.
حالا جداکننده حذف و بقیه خط سنجیده می‌شود، و شمارش کلمه با سقف دوازده (به‌جای
نه) برگشت تا سرفصل‌های بلندِ نقطه‌گذاری‌شده حفظ شوند. قاعده عدد نقطه‌دار سقف
تنگ‌تر خودش (نه) را دارد چون نقطه‌گذاری‌ای برای خواندن ندارد. هر ۲۳ نمونه
(۱۰ متن، ۱۳ سرفصل) درست دسته‌بندی می‌شوند.

**سه کران در بسته‌بندی رعایت نمی‌شد.** وقتی کل محتوای در انتظار سرفصل بود،
جداسازی همه‌اش را برمی‌داشت و بلافاصله برمی‌گرداند، پس کران `size` هرگز اعمال
نمی‌شد: ۲۹ سرفصل پیاپی یک چانک ۵۱۱ نویسه‌ای می‌ساخت. ادغام سرفصل انتهای سند هم
بدون بررسی طول به چانک آخر می‌چسبید. هر دو حالا کران را رعایت می‌کنند.

سرفصل‌هایی که از نصف `size` بلندتر بودند جدا emit می‌شدند، که همان چانک بی‌متنی
را می‌ساخت که این کد برای جلوگیری از آن نوشته شده. حالا همه سرفصل‌ها وارد
پنجره‌بندی بخش بعدی می‌شوند و هر پنجره بر اساس چیزی که واقعاً دارد برچسب
می‌خورد.

**شماره صفحه از سهم غالب پنجره می‌آید.** پنجره‌ای با ۱۱ نویسه سرفصل از صفحه یک
و ۲۸۹ نویسه متن از صفحه دو، صفحه یک ارجاع داده می‌شد. حالا هر پنجره به صفحه‌ای
ارجاع می‌دهد که بیشتر متنش از آنجاست — که در جهت عکس هم درست کار می‌کند: پنجره‌ای
که عمدتاً متن حمل‌شده است، صفحه همان را نگه می‌دارد.


## بازبینی هفتم: بازسازی به‌جای وصله هفتم

شش دور اصلاح روی `chunk_blocks`، هر بار حالت خاص تازه‌ای اضافه کرد و هر بار
بازبینی بعدی ترکیب تازه‌ای پیدا کرد که آن حالت‌ها پوششش نمی‌دادند. آخرینش:
سرفصلی که متن زیرش کنارش جا نمی‌شد، دوباره تنها می‌ماند.

تابع حول یک ایده بازنویسی شد: **سرفصل پیش از شروع بسته‌بندی به متنی که باز
می‌کند چسبانده می‌شود.** `_units` هر دنباله سرفصلِ بی‌متن را با بلوک بعدی یکی
می‌کند (و دنباله انتهای سند را با واحد قبلی). بسته‌بندی بعد از آن فقط واحدهایی
می‌بیند که خودشان متن دارند، پس هیچ ترتیبی از اندازه‌ها نمی‌تواند سرفصلی را
تنها بگذارد — نه با حالت خاص، بلکه با ساختار.

سه شاخه ویژه حذف شدند: جداسازی سرفصل‌های انتهایی، شاخه «اگر چیزی جز سرفصل
نمانده»، و ادغام کران‌دار انتهای سند. برچسب‌گذاری پنجره‌ها هم دقیق شد: هر بلوک
بازه مشخصی از متن واحد را اشغال می‌کند، پس پنجره فقط بخش‌هایی را اعلام می‌کند
که واقعاً با آن‌ها هم‌پوشانی دارد و به صفحه‌ای ارجاع می‌دهد که بیشترین نویسه را
داده است.

سه یافته دیگر بازبینی بازتولید نشدند (پنجره سرفصل‌ها زیر نام بخش بدنه، سرفصل
انتهایی تنها، و کران اندازه در حالت سرفصل+بدنه) و یافته مربوط به سقف دوازده
کلمه بر پایه فرض غلطی بود: با آزمون مستقیم، `_titleish_tail` هنوز
`Chapter 4 - was written by the finance team in the last quarter of the year`
را عنوان می‌داند، پس سقف لازم است.

اعتبارسنجی: ۴۰۰ سند تصادفی با اندازه و هم‌پوشانی متغیر، بدون افتادن هیچ واژه،
بدون عبور از `size`، و با برگ بخش همیشه میان بخش‌های اعلام‌شده. همین بررسی به
شکل یک آزمون ماندگار (۲۰۰ سند) اضافه شد.
