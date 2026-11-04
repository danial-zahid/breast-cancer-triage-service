
# =========================
# STAGE 1: BUILDER
# =========================
FROM python:3.11.11-slim-bookworm AS builder

WORKDIR /build

# Build arguments
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

# Install training dependencies first
COPY requirements-train.txt .

RUN pip install --no-cache-dir -r requirements-train.txt

# Copy dataset and training-related files
COPY data/raw/breast_cancer.csv data/raw/breast_cancer.csv
COPY training/ training/
COPY params.yaml .

# Train model during builder stage
RUN python training/train.py \
    --data-path data/raw/breast_cancer.csv \
    --output-dir /build/artifacts \
    --model-version ${MODEL_VERSION}


# =========================
# STAGE 2: PRODUCTION
# =========================
FROM python:3.11.11-slim-bookworm AS production

WORKDIR /app

# Build arguments
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

# Runtime environment variables
ENV MODEL_PATH=/app/artifacts/model.joblib
ENV MODEL_VERSION=${MODEL_VERSION}
ENV GIT_SHA=${GIT_SHA}
ENV DECISION_THRESHOLD=0.5
ENV LOG_LEVEL=INFO
ENV REDIS_HOST=""
ENV REDIS_PORT=6379
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only serving dependencies
COPY requirements-serve.txt .

RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy API code
COPY app/ app/

# Copy only trained model artifacts from builder
COPY --from=builder /build/artifacts/model.joblib artifacts/model.joblib
COPY --from=builder /build/artifacts/model_card.json artifacts/model_card.json

# Create non-root user
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup appuser \
    && chown -R appuser:appgroup /app

USER appuser

# Image metadata
LABEL maintainer="ML-ops Assignment"
LABEL description="Breast Cancer Triage Prediction API"
LABEL version="${MODEL_VERSION}"

EXPOSE 8000

# Health check using Python standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Start API
ENTRYPOINT ["python", "-m", "uvicorn"]

CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
