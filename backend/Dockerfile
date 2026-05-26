# syntax=docker/dockerfile:1.7
#
# PrepBuddy backend — Hugging Face Spaces (Docker SDK)
#
# Why this layout:
#   * HF Docker Spaces expect the app to listen on $PORT (default 7860).
#   * Free tier gives 16 GB RAM and ~50 GB disk, so we pre-download the models
#     at build time. That keeps the first user request fast (no 30-60s cold
#     start while sentence-transformers, DeBERTa NLI, and spaCy are fetched).
#   * Storage on the free tier is ephemeral (resets on restart). The session
#     and LLM-cache SQLite files still work fine for a demo; document this in
#     DEPLOYMENT.md.
#   * Multi-stage build keeps the final image close to runtime-only deps.

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    TORCH_HOME=/app/.cache/torch

# Minimal system deps. libgomp1 is needed by PyTorch on CPU; curl is for
# the HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- deps stage
FROM base AS deps
WORKDIR /app

# Install Python deps first so they cache independently of the source code.
COPY requirements.txt ./
# Install CPU-only torch from the official index to avoid pulling the much
# larger CUDA wheels (saves ~2 GB).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Pre-download spaCy English model so it is available offline at runtime.
RUN python -m spacy download en_core_web_sm

# ---------------------------------------------------------------- model warm-up
# Pre-fetch the sentence-transformers and NLI weights into the cache directory
# so the first request after deploy does not pay the download cost.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small")
AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/nli-deberta-v3-small"
)
print("Pre-warmed SBERT + NLI weights.")
PY

# ---------------------------------------------------------------- runtime stage
FROM base AS runtime

# Hugging Face Spaces run containers as a non-root user with UID 1000. Create
# a matching user so writes to /app/data and /app/.cache work.
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy installed Python packages and the warmed model cache.
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY --from=deps /app/.cache /app/.cache

# Copy the application code.
COPY --chown=user:user . /app

# HF Spaces will pass PORT; default to 7860 for local docker run.
ENV PORT=7860
EXPOSE 7860

USER user

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

# Bind to 0.0.0.0 so the container is reachable from outside.
# Use a single worker — the models stay loaded in this process, and HF
# Spaces free tier has limited RAM if we fork multiple workers.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
