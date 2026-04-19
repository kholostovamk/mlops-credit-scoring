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

# Copy source code
COPY src/           ./src/
COPY inference/     ./inference/
COPY models/        ./models/   2>/dev/null || true

# SageMaker expects the serve script at /opt/program/serve
COPY inference/predict.py ./predict.py

# SageMaker serve entrypoint
ENV SAGEMAKER_PROGRAM predict.py
ENV SM_MODEL_DIR /opt/ml/model

# Expose port for local testing (Flask)
EXPOSE 8080

# Default: run the inference server
CMD ["python", "-c", "\
import os, json, joblib; \
from flask import Flask, request, jsonify; \
from inference.predict import model_fn, input_fn, predict_fn, output_fn; \
app = Flask(__name__); \
MODEL = model_fn(os.environ.get('SM_MODEL_DIR', 'models')); \
@app.route('/ping', methods=['GET']); \
def ping(): return 'OK', 200; \
@app.route('/invocations', methods=['POST']); \
def invoke(): \
    ct = request.content_type or 'application/json'; \
    inp = input_fn(request.get_data(as_text=True), ct); \
    out = predict_fn(inp, MODEL); \
    body, mime = output_fn(out, request.headers.get('Accept', 'application/json')); \
    return body, 200, {'Content-Type': mime}; \
app.run(host='0.0.0.0', port=8080) \
"]
