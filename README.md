# AdaptiveMetric RAG

**Not every query should use the same notion of similarity.**

AdaptiveMetric RAG is a self-hosted, multilingual knowledge assistant that changes its retrieval strategy for every question. A temporal question emphasizes dates, a factual question emphasizes entities and exact terms, and a conceptual question emphasizes semantic similarity. Every answer includes inspectable source citations.

[راهنمای فارسی](README.fa.md)

## What is included

- Adaptive query analyzer with factual, conceptual, causal, numeric, temporal, and technical/code intents
- Per-query mixture of dense, BM25, entity, numeric, temporal, and metadata scores
- Multilingual multi-keyword expansion with blended query vectors for cross-language retrieval
- Three-stage flow: fast candidate selection → adaptive scoring → grounded generation
- Confidence scoring and early-exit signals
- PDF, DOCX, TXT, Markdown, CSV, JSON, and HTML ingestion
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
Query → Query Analyzer → Metric Router
                            │
          Dense + BM25 + Entity + Number + Time + Metadata
                            │
                    Candidate pool (10–500)
                            │
                  Adaptive metric scoring
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

API keys submitted through the interface are stored server-side and never returned to the browser. For public or multi-user production deployments, inject secrets through environment variables or a secrets manager and place the application behind authentication and TLS.

## Retrieval settings

| Setting | Default | Purpose |
|---|---:|---|
| Candidate pool | 100 | Number of fused dense/BM25 candidates evaluated by the adaptive metric |
| Context chunks | 5 | Sources passed to the answer provider |
| Chunk size | 900 | Approximate characters per chunk |
| Chunk overlap | 140 | Character overlap between adjacent chunks |
| Confidence threshold | 0.58 | Threshold exposed for confidence-aware flows |
| Early exit | On | Marks decisive retrievals so expensive optional stages can be skipped |
| Query expansion | On | Enables expansion behavior in confidence-aware extensions |

Chunk settings apply to newly uploaded documents. Re-upload existing documents after changing them.

Uploading a file whose contents are already in the library is rejected; delete the
existing document first to replace it.

## Measuring retrieval quality

```bash
python -m eval.run_eval --verbose          # score the golden set
python -m eval.run_eval --embedding gemini --embedding-model gemini-embedding-001
python -m eval.bench_latency               # latency at 1k / 5k / 20k chunks
pytest -m eval                             # the same run as a test
```

`eval/` holds a 13-document Persian and English corpus and 89 labelled questions
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
