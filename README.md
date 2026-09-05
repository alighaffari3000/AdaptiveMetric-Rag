# AdaptiveMetric RAG

**Not every query should use the same notion of similarity.**

AdaptiveMetric RAG is a self-hosted, multilingual knowledge assistant that changes its retrieval strategy for every question. A temporal question emphasizes dates, a factual question emphasizes entities and exact terms, and a conceptual question emphasizes semantic similarity. Every answer includes inspectable source citations.

[راهنمای فارسی](README.fa.md)

## What is included

- Adaptive query analyzer with factual, conceptual, causal, numeric, temporal, and technical/code
  intents, scored independently so one question can be several of them at once
- Conversation-aware rewriting: a follow-up such as "and its cost?" is made standalone from the
  recent turns before retrieval, offline by default and by the answer model when one is configured
- One Persian text normalizer behind ingestion, BM25 and every lexical signal: Arabic ی/ک, the half
  space, kashida, harakat, and ۱۴۰۳/١٤٠٣/1403 all compare equal, and ۱۵ خرداد ۱۴۰۲ matches ۱۴۰۲/۰۳/۱۵
- Per-query mixture of dense, BM25, entity, numeric, temporal, and metadata signals,
  fused by rank so no signal wins on the scale it happens to use
- Optional second-stage reranking by a local cross-encoder or by the answer model
- Multilingual multi-keyword expansion with blended query vectors for cross-language retrieval
- Three-stage flow: fast candidate selection → adaptive scoring → grounded generation
- Confidence scoring and early-exit signals
- Server-sent streaming at `/api/chat/stream`: the sources appear before the model starts writing,
  and repair happens after the stream rather than rewriting text already on screen
- Every cited sentence is checked against the passage it cites, and the ones the passage does not
  support are counted and marked rather than silently trusted
- Optional structured citations: the model returns `{answer, claims}` and it is rendered back into
  inline `[1]` markers, falling back to reading the markers out of prose
- PDF, DOCX, TXT, Markdown, CSV, JSON, and HTML ingestion, with headings kept as a section path
  and table rows serialised one per line
- Persian PDFs read in the order they were written: mirrored pages are detected and repaired, and
  numbers a bidi pass reversed are restored by cross-checking the two PDF engines
- Parent/child chunking: small chunks are retrieved, the window around them is what the model reads,
  and a chunk may cross a page break so a fact split by one survives
- Ingestion runs behind the request, with progress at `/api/documents/{id}/status`
- Inline citations with source excerpts, page numbers, chunk IDs, and retrieval scores
- Local no-key extractive mode, Ollama, OpenAI-compatible/dedicated endpoints, and Google Gemini
- Semantic multilingual embeddings from a local sentence-transformers model, Ollama, an
  OpenAI-compatible endpoint, or Gemini, with automatic re-indexing when the model changes
- A golden-set evaluation harness reporting Recall@K, MRR, nDCG and abstention accuracy
- Persistent conversations, documents, and settings in SQLite
- Responsive Persian-first UI with a knowledge library and retrieval diagnostics
- Rich Markdown answer rendering plus persistent dark and light themes
- Per-conversation JSON export with messages, citations, and safe runtime metadata
- Single-command Docker deployment on port `2266`

## Architecture

```text
Query + recent turns → Rewriter → Query Analyzer → Metric Router
                            │
          Dense + BM25 + Entity + Number + Time + Metadata
                            │
                    Candidate pool (10–500)
                            │
              Adaptive fusion of the ranked signals
                            │
             Cross-encoder or LLM rerank (optional)
                            │
              Confidence / early-exit decision
                            │
                 Top grounded context chunks
                            │
             Local / Ollama / OpenAI / Gemini
                            │
                    Answer + citations
```

Retrieval is served from an in-memory index built once at startup and refreshed on
ingest, delete, and re-index: a float32 matrix for dense scoring and an inverted token
index for BM25. Retrieval over 20,000 chunks takes about 13 ms.

### Choosing an embedding

The zero-setup default is 384-dimensional feature hashing. It starts instantly and
runs offline, but it is a **lexical** signal, not a semantic one: a Persian question
and its English answer share no features, so cross-language retrieval does not work.
Use it for a quick trial or a single-language corpus.

For real use, pick a semantic model under **Settings -> Retrieval**:

| Provider | Suggested model | Notes |
|---|---|---|
| `sentence-transformers` | `intfloat/multilingual-e5-small` | Fully local, about 470 MB, CPU friendly |
| `sentence-transformers` | `BAAI/bge-m3` | Higher accuracy, about 2.2 GB |
| `ollama` | `bge-m3` | Uses a model already pulled in Ollama |
| `openai` | `text-embedding-3-small` | Any OpenAI-compatible `/embeddings` endpoint |
| `gemini` | `gemini-embedding-001` | 768 dimensions, strong multilingual quality |

The `sentence-transformers` option needs the optional extra:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-local-embeddings.txt
```

For Docker, build with `WITH_LOCAL_EMBEDDINGS=1` to bundle it (about 1 GB extra).
Models are cached in the data volume, so a container rebuild does not re-download them.

Document vectors include the filename, section, and chunk content. With a semantic
model the question is embedded as written; feature hashing keeps the keyword-expanded
variant blending it needs to match anything across languages.

## Quick start with Docker

```bash
git clone https://github.com/alipyth/AdaptiveMetric-Rag
cd AdaptiveMetric-RAG
docker compose up --build -d
```

Open **http://localhost:2266**. The default local provider requires no API key.

To stop the service:

```bash
docker compose down
```

Data is stored in the named Docker volume `adaptive_rag_data` and survives container recreation.

## Provider setup

Open **Settings → Model provider** in the UI.

Generation and embedding are configured independently. Under **Settings → Retrieval**, choose the built-in local 384-dimensional feature hashing or select an embedding model already installed in Ollama. The UI can load installed Ollama models and automatically re-embeds all existing chunks before activating a changed model.

### Ollama

1. Run Ollama on the host and pull a model, for example `ollama pull llama3.2`.
2. Select **Ollama**.
3. Use model `llama3.2` and URL `http://host.docker.internal:11434`.

### OpenAI or a dedicated OpenAI-compatible endpoint

Select **OpenAI / Dedicated compatible**, enter the model, API key, and base URL. The default is `https://api.openai.com/v1`; a vLLM, LM Studio, corporate gateway, or dedicated endpoint can be used if it implements `POST /chat/completions`.

You can alternatively provide `OPENAI_API_KEY` in `.env`.

### Google Gemini

Select **Google Gemini**, enter a Gemini model such as `gemini-2.5-flash`, and add the API key. You can alternatively provide `GEMINI_API_KEY` in `.env`.

API keys submitted through the interface are stored server-side and never returned to the browser.
Set `APP_SECRET_KEY` to encrypt them at rest, or `APP_ENV=production` to read them from the
environment only and have the API refuse to store one at all.

| Variable | Effect |
|---|---|
| `APP_SECRET_KEY` | Encrypts saved API keys at rest |
| `APP_ENV=production` | Keys come from the environment; saving one returns 403 |
| `APP_AUTH_TOKEN` | Every request except `/health` needs `Authorization: Bearer <token>` |
| `APP_RATE_LIMIT_CHAT` / `APP_RATE_LIMIT_UPLOAD` | Requests per minute per client; `0` disables |
| `APP_LOG_FORMAT` / `APP_LOG_LEVEL` | `json` (default) or `text`; `INFO` by default |

Logs are one JSON object per line carrying the request id, which is taken from `X-Request-ID`
when the caller sends one and returned on every response. Still place a public deployment behind
TLS: a bearer token over plain HTTP is a token anyone on the path can read.

## Retrieval settings

| Setting | Default | Purpose |
|---|---:|---|
| Candidate pool | 100 | Number of fused dense/BM25 candidates evaluated by the adaptive metric |
| Context chunks | 5 | Sources passed to the answer provider |
| Child tokens | 250 | Size of the chunks the index scores |
| Child overlap | 40 | Overlap between adjacent chunks, in tokens |
| Parent tokens | 900 | Size of the window handed to the answer provider |
| Semantic signal weight | 0.85 | Share of the ranking the embedding gets when it is semantic |
| Rerank | Off | Second-stage reranking of the shortlist |
| Rerank shortlist | 30 | Candidates handed to the reranker |
| Rerank weight | 0.7 | Share of the final score from the reranker; the rest is the fusion score |
| Confidence threshold | 0.58 | Threshold exposed for confidence-aware flows |
| Early exit | On | Marks decisive retrievals so expensive optional stages can be skipped |
| Query expansion | On | Enables expansion behavior in confidence-aware extensions |
| Query rewrite | On | Makes a follow-up question standalone from the conversation before retrieval |
| Verify claims | On | Checks each cited sentence against the passage it cites |
| Verify backend | lexical | `lexical` is offline and free; `llm` asks the answer model instead |
| Structured citations | Off | Asks the model for `{answer, claims}` instead of inline markers only |
| Multi-query | On | Retrieves the model's alternative phrasings too and merges them by rank |
| History turns | 4 | Messages of the conversation the rewriter and the answer prompt may see |
| Strict multi-part answers | Off | Treats a short answer to a two-part question as truncated and repairs it |

Chunk settings apply to newly uploaded documents. Re-upload existing documents after changing them.
Token counts are estimated at four characters per token for Latin text and three for Persian.

Scanned pages are sent to `tesseract -l fas+eng` when that binary is installed; without it the
upload status carries a warning instead of silently indexing an empty page.

Uploading a file whose contents are already in the library is rejected; delete the
existing document first to replace it.

## Measuring retrieval quality

```bash
python -m eval.run_eval --verbose          # score the golden set
python -m eval.run_eval --embedding gemini --embedding-model gemini-embedding-001
python -m eval.bench_latency               # latency at 1k / 5k / 20k chunks
pytest -m eval                             # the same run as a test
```

`eval/` holds a 13-document Persian and English corpus and 98 labelled questions
covering cross-lingual, multi-intent, follow-up, page-boundary and unanswerable
cases. Recorded numbers for each change live in `eval/BASELINE.md`. See
`eval/README.md` for the format and how to add cases.

## Local development

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 2266
```

Run tests:

```bash
pytest -q
```

API documentation is available at `http://localhost:2266/docs`.

## Citation behavior

Retrieved chunks are numbered before generation. The model is instructed to cite factual claims as `[1]`, `[2]`, etc. The API also returns a structured `citations` array independently of model formatting, so clients can always display the source document, page, excerpt, chunk ID, and adaptive score.

## API example

```bash
curl -X POST http://localhost:2266/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"When does contract 137 expire?"}'
```

## Production notes

- Put the service behind an authenticated reverse proxy before exposing it publicly.
- SQLite is deliberately simple for a single-node deployment. Use PostgreSQL for multi-user writes.
- For very large collections, use Qdrant/FAISS for first-stage ANN retrieval and retain adaptive scoring over the top 50–200 candidates.
- Add OCR before ingestion for scanned PDFs; `pypdf` extracts text but does not perform OCR.
- Benchmark retrieval on your own labeled query/chunk pairs using Recall@K, MRR, nDCG, and p95 latency.

## License

Add the license appropriate for your intended distribution before publishing the repository.
