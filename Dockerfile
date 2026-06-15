# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Remove .pyc bytecode files and __pycache__ directories
    find /opt/venv -name '*.pyc' -delete && \
    find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    # Remove pip's persistent cache in /root
    rm -rf /root/.cache/pip 2>/dev/null || true

# Notes:
# - docling + torch/torchvision REMOVED (2026-06-07): The previous implementation
#   used docling + torch for ML-powered PDF layout analysis and OCR, which consumed
#   ~1 GB RAM and caused OOM kills on t2.micro (1 GB). Replaced with pypdf for
#   lightweight text-only extraction — no ML models, no torch, no OpenCV deps.
#   See git history for the original implementation.


# Production stage
FROM python:3.12-slim AS production

WORKDIR /app

# GIT_COMMIT_SHA is baked into the image for deployment verification.
# Pass via --build-arg GIT_COMMIT_SHA=$(git rev-parse HEAD).
# The .env file on EC2 also sets this as a runtime fallback.
ARG GIT_COMMIT_SHA=unknown
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -m -g appgroup appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY app/ ./app/
COPY business_rules/ ./business_rules/

RUN chown -R appuser:appgroup /app /opt/venv

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
