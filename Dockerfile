FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/app/data \
    HF_HOME=/app/data/models
WORKDIR /app

# Build with --build-arg WITH_LOCAL_EMBEDDINGS=1 to bake in sentence-transformers
# and a CPU-only torch. That adds roughly 1 GB to the image; without it the
# service still runs, and semantic embeddings come from Ollama, an
# OpenAI-compatible endpoint, or Gemini instead.
ARG WITH_LOCAL_EMBEDDINGS=0

COPY requirements.txt requirements-local-embeddings.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$WITH_LOCAL_EMBEDDINGS" = "1" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements-local-embeddings.txt; \
    fi

COPY app ./app
COPY static ./static
COPY README.md README.fa.md ./
RUN mkdir -p /app/data/uploads /app/data/models

EXPOSE 2266
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:2266/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2266"]
