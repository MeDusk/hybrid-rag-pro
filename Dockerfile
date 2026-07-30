# ============================================================
# HybridRAG-Pro — Multi-stage Dockerfile
# Python 3.11 slim — production-optimized
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building native extensions (faiss, transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY .env.example .env.example

# Create necessary data directories
RUN mkdir -p \
    ./data/faiss_index \
    ./data/uploads \
    ./eval_results \
    ./logs \
    ./mlruns

# Non-root user for security
RUN addgroup --system hybridrag && \
    adduser --system --ingroup hybridrag hybridrag && \
    chown -R hybridrag:hybridrag /app

USER hybridrag

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
