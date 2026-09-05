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

# Phase 4 — question understanding and conversation memory

Measured 2026-09-05 on the same corpus and golden set. The semantic
configuration could not be re-measured: this machine has no embedding API key,
so every number here is the feature-hashing default, which is also the
configuration Phases 0 and 1 were recorded on.

Reproduce with:

```
python -m eval.run_eval --rewrite off     # normalizer and router only
python -m eval.run_eval                   # the default: offline rewriter as well
```

## Quality

| Metric | Before Phase 4 | Normalizer + router | With the rewriter |
|---|---:|---:|---:|
| hit@5 | 0.800 | 0.811 | **0.844** |
| recall@5 | 0.794 | 0.806 | **0.839** |
| MRR | 0.709 | 0.710 | **0.743** |
| nDCG@10 | 0.736 | 0.737 | **0.770** |
| answerable not abstained | 0.956 | 0.956 | **0.978** |
| latency p95 (ms) | 2.2 | 1.8 | 1.7 |

By tag, recall@5:

| Tag | Before | After |
|---|---:|---:|
| follow_up | 0.500 | **1.000** |
| fa2fa | 0.826 | **0.919** |
| numeric | 0.726 | **0.823** |
| causal | 0.714 | **0.786** |
| en2en | 1.000 | 1.000 |
| cross_lingual | 0.400 | 0.400 |

The acceptance criterion for the phase was recall@5 above 0.7 on the follow-up
subset. All four cases now retrieve their document.

## Where the gain comes from

**The normalizer, not the router.** Folding Persian text once - Arabic `ي`/`ك`,
the half space, kashida, harakat, and `۱۴۰۳`/`١٤٠٣`/`1403` - is what moved
`fa2fa` and `numeric`. Before it, `DATE_RE` could not match `۱۴۰۲` at all,
because the `1` and the `4` in its year pattern were Latin literals: every
Persian date in the corpus was invisible to the temporal signal, which is the
signal that intent weighting leans on hardest for those questions. Numbers now
compare as canonical values rather than as substrings, so `۲٬۴۰۰٬۰۰۰٬۰۰۰`
matches `2,400,000,000` while `137` no longer matches `1370`.

**The rewriter carries the whole follow-up gain.** Retrieval on «و مبلغش چقدر
بود؟» has nothing to work with; the same question with the previous turn's topic
appended reaches the right contract every time. It also lifted
`answerable_not_abstained`: a question the retriever cannot ground is a question
the evidence gate refuses.

**The multi-label router is close to quality-neutral here.** Splitting a
question's vote across the intents it actually states moved the aggregate by
about one point, and `multi_intent` did not move at all. That is the honest
result: with feature hashing the dense signal carries no meaning, so any mix of
weights is a mix of lexical signals. The router's value is that
«چرا مبلغ قرارداد ۱۳۷ در سال ۱۴۰۳ تغییر کرد؟» is now scored as causal, numeric
and temporal at once instead of being filed as temporal and having the rest of
the sentence discarded - which is what a semantic embedding needs in order to
put its weight where the question is. It should be re-measured on the semantic
configuration before the effect is claimed either way.

## What did not move

`cross_lingual`, `fa2en` and `en2fa` are unchanged, as expected: no amount of
folding makes a Persian token match an English one. Only a semantic embedding
does, and Phase 2 already measured that (0.900 cross-lingual recall with
`gemini-embedding-001`). The two `page_boundary` cases stay at zero until
Phase 5. `abstain_accuracy` stays at 0.125, unchanged since Phase 0 and still
the weakest number in the suite.

## Multi-query is implemented but unmeasured

Alternative phrasings come from the LLM rewriter, which needs a provider. The
fusion itself is covered by unit tests and the rule-based path produces no
variants, so every number above is single-query. `--rewrite llm` measures the
other path when a key is available.

## Cost

The rewriter adds one cheap completion per question for non-local providers,
cached by conversation, message and the turns it was derived from. The offline
path adds no call at all and costs about 0.1 ms. Folding each chunk at index
build time is what keeps the query path free of it: p95 latency did not rise.

---

# Phase 5 — document input and chunking

Measured 2026-09-05, feature-hashing default, same corpus and golden set.

Reproduce with:

```
python -m eval.run_eval
python -m eval.run_eval --child-tokens 500      # the sweep below
pytest tests/test_ingestion.py
```

## Quality

| Metric | Phase 4 | Phase 5 |
|---|---:|---:|
| hit@5 (child chunks) | 0.844 | 0.833 |
| recall@5 | 0.839 | 0.830 |
| MRR | 0.743 | 0.729 |
| nDCG@10 | 0.770 | 0.759 |
| **delivered hit@5** | 0.844 | **0.878** |
| page_boundary hit@5 | 0.000 | **1.000** |
| pdf hit@5 | 0.500 | **1.000** |
| answerable not abstained | 0.978 | 0.978 |

`delivered_hit@5` is new and is the honest measure of this phase. The ranking
metrics score child chunks; the answering model is handed the parent window
around each retrieved child, so a fact one sentence past the child's edge still
reaches it. In Phase 4 there were no parents, so the two numbers were the same
thing. Reported separately, never mixed into hit@5.

Read together: the child-level numbers dipped by about one point because the
retrieval task got harder - 31 chunks became 32, but they now follow sections
rather than character counts, so a question about the training budget has to
pick the right third of the HR policy instead of hitting the one chunk that was
the whole document. What actually reaches the model improved by 3.4 points.

## The two page-boundary cases

Both now pass, which was the phase's acceptance criterion. The fix that
mattered was not page-spanning chunks by themselves but the overlap rule:
consecutive children carry over the previous sentence even when it is longer
than the overlap budget. The evaluation PDF's cold-chain explanation is a
60-token sentence cut by the page break, and the budget is 40, so the carry-over
was skipped and both halves lost the other. One sentence is now always carried
when it fits in half a chunk.

## Persian PDFs: the plan's premise did not hold

The plan assumed PyMuPDF reads Persian in the correct order and `pypdf` is the
fallback. Measured on two fixtures built with the same text and opposite layout
strategies (`eval/make_fixtures.py`, `tests/test_ingestion.py`), comparing
codepoints rather than what a terminal displays:

| Producer laid glyphs out | PyMuPDF | pypdf |
|---|---|---|
| right-to-left (bidi-aware) | words correct, **digit runs reversed** (۱۳۷ read as ۷۳۱) | fully correct |
| left-to-right (naive) | every word mirrored | every word mirrored, word order flipped too |

So neither engine is right on its own. What ships instead: PyMuPDF reads
structure; mirrored pages are detected by counting common Persian words against
their reversals and repaired by un-mirroring; and reversed numbers are repaired
by cross-checking each run against the other engine, which only rewrites a run
when the reference holds exactly its reversal. Both fixtures now extract every
Persian word, `۱۳۷`, `۲٬۴۰۰٬۰۰۰٬۰۰۰` and `۱۴۰۶/۰۱/۱۴` intact.

The word-count detector had to be whole-word: counting substrings let «را» and
«در» match by accident inside longer words in both directions, so a short
mirrored line scored zero and went unrepaired.

## Chunk size: what the sweep says, and why the default ignores it

| child tokens | chunks | hit@5 | recall@5 | MRR | nDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 180 | 61 | 0.833 | 0.822 | 0.710 | 0.739 |
| 250 (default) | 32 | 0.833 | 0.830 | 0.729 | 0.759 |
| 320 | 54 | 0.844 | 0.833 | 0.728 | 0.762 |
| 400 | 54 | 0.844 | 0.839 | 0.730 | 0.761 |
| 500 | 50 | 0.867 | 0.856 | 0.749 | 0.778 |
| 650 | 49 | 0.867 | 0.861 | 0.750 | 0.781 |

(The 320-650 rows were measured before small sections were packed together,
which is why their chunk counts are higher than the default's.)

Quality rises with chunk size all the way up, and the default stays at 250
anyway. The reason is that the sweep cannot answer the question it appears to
answer. The default embedding is feature hashing, a lexical signal: a longer
chunk contains more terms and therefore matches more questions, without being
a better retrieval unit. At the limit the winning strategy is one chunk per
document, which is what the corpus had before this phase and why its hit@5 was
so high. A semantic embedding behaves the opposite way - a long chunk averages
several topics into one vector and blurs it - and that is the configuration
this project is aimed at. Tuning the chunk size on the fallback embedding would
optimise for the one setup where the answer is "make chunks bigger".

`child_tokens`, `child_overlap_tokens` and `parent_tokens` are settings, so a
library that stays on feature hashing can raise the first one.

## Token counting

Chunk sizes are in tokens now rather than characters, estimated as four
characters per token for Latin text and three for Persian, counting spaces.
It is an estimate, not a tokenizer, and it only has to be stable - every size
above was chosen by measuring with this counter. A real tokenizer would move
the numbers a little and none of the conclusions.

## Also in this phase

- Ingestion runs behind the request. A large PDF used to hold the HTTP
  connection open for its whole extraction and embedding run; the document row
  now appears immediately as `processing` and `/api/documents/{id}/status`
  reports progress, warnings and failures.
- OCR is a hook, not a dependency: a page with no text layer is sent to
  `tesseract -l fas+eng` when the binary exists, and otherwise produces a
  warning in the upload status rather than silently indexing nothing. No
  tesseract binary was available on this machine, so that path is implemented
  and unmeasured.
- CSV and DOCX tables serialise one line per row with each cell labelled by its
  column, so a row survives chunking as a unit.
- Citations carry a page range and the full section path, both of which exist
  only now that a chunk can span pages and knows its heading.
