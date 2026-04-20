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

# Create a 'serve' executable that SageMaker invokes as: docker run <image> serve
# SageMaker overrides CMD with 'serve', so this must exist as a runnable command in PATH.
RUN printf '#!/bin/bash\nexec python /opt/program/serve.py "$@"\n' > /opt/program/serve \
    && chmod +x /opt/program/serve

# Model is baked into the image at /opt/program/models
ENV SM_MODEL_DIR=/opt/program/models \
    PATH="/opt/program:${PATH}"

# Expose port for local testing and SageMaker
EXPOSE 8080

# Default CMD for local testing; SageMaker overrides this with 'serve'
CMD ["serve"]
