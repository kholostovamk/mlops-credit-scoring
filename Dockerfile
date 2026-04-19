# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder – install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for xlrd / XGBoost compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime – minimal image for inference
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp needed at runtime by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /opt/program

# Copy source code, inference scripts, and baked-in model artefacts.
# models/ is populated by the CI artifact download step before docker build.
COPY src/               ./src/
COPY inference/         ./inference/
COPY models/            ./models/

# Bring predict.py and serve.py to the working directory root so imports work
COPY inference/predict.py ./predict.py
COPY inference/serve.py   ./serve.py

# Model is baked into the image at /opt/program/models
ENV SM_MODEL_DIR /opt/program/models

# Expose port for local testing and SageMaker
EXPOSE 8080

# Start the Flask inference server
CMD ["python", "serve.py"]
