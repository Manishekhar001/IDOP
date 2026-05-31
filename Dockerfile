# Build stage
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Notes:
# - CPU-only PyTorch: torch is installed FIRST from the CPU-only index, so CUDA
#   runtime packages (~1.5-2GB) never enter the venv. EC2 t2.micro has no GPU.
# - torch has been removed from requirements.txt to prevent PyPI pulling in CUDA torch.
# - docling uses torch at runtime for PDF parsing — CPU-only is sufficient.


# Production stage
FROM python:3.13-slim AS production

WORKDIR /app

# GIT_COMMIT_SHA is baked into the image for deployment verification.
# Pass via --build-arg GIT_COMMIT_SHA=$(git rev-parse HEAD).
# The .env file on EC2 also sets this as a runtime fallback.
ARG GIT_COMMIT_SHA=unknown
ENV GIT_COMMIT_SHA=$GIT_COMMIT_SHA

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY app/ ./app/
COPY business_rules/ ./business_rules/

RUN chown -R appuser:appgroup /app

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
