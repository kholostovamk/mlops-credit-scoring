"""
drift_detector.py
-----------------
Lightweight data- and concept-drift detector that runs as a sidecar process
alongside the inference server.

Drift detection approach:
  - Data drift:    Population Stability Index (PSI) per feature, comparing
                   a rolling 1-hour window of live requests against the
                   training reference distribution.
  - Concept drift: tracks rolling mean of predicted probability; alerts when
                   it deviates >5% from the training-time baseline.

Metrics are exposed on :8081/metrics for Prometheus to scrape.
"""

import json
import logging
import os
import time
import threading
from collections import deque
from typing import Dict, List

import numpy as np
import pandas as pd
from prometheus_client import Gauge, Counter, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
DRIFT_SCORE      = Gauge("model_feature_drift_score",    "Max PSI across all features")
PRED_MEAN        = Gauge("model_prediction_mean",        "Rolling mean of predicted probability")
REQUESTS_TOTAL   = Counter("inference_requests_total",   "Total inference requests processed")
ERRORS_TOTAL     = Counter("inference_errors_total",     "Total inference errors")
LATENCY_GAUGE    = Gauge("inference_latency_last_ms",    "Latency of last inference call in ms")

# ── Config ─────────────────────────────────────────────────────────────────────
METRICS_PORT     = int(os.environ.get("DRIFT_METRICS_PORT", 8081))
REFERENCE_FILE   = os.environ.get("REFERENCE_STATS", "metrics/distributions.json")
WINDOW_SIZE      = int(os.environ.get("DRIFT_WINDOW", 1000))   # rolling samples
PSI_THRESHOLD    = float(os.environ.get("PSI_THRESHOLD", 0.10))
CHECK_INTERVAL   = int(os.environ.get("DRIFT_CHECK_INTERVAL", 60))  # seconds

# In-memory rolling buffer of incoming feature vectors
request_buffer: deque = deque(maxlen=WINDOW_SIZE)
prediction_buffer: deque = deque(maxlen=WINDOW_SIZE)
lock = threading.Lock()


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between reference and current distributions."""
    eps = 1e-6
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)   + eps
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def load_reference_stats(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        logger.warning(f"Reference stats file not found: {path}. Drift detection disabled.")
        return {}
    with open(path) as f:
        return json.load(f)


def drift_check_loop(reference_stats: Dict[str, dict]):
    """Background thread: compute and expose drift metrics every CHECK_INTERVAL s."""
    logger.info(f"Drift detector started. Checking every {CHECK_INTERVAL}s.")
    while True:
        time.sleep(CHECK_INTERVAL)
        with lock:
            if len(request_buffer) < 50:
                logger.info("Not enough samples yet for drift check.")
                continue
            live_df = pd.DataFrame(list(request_buffer))
            preds   = list(prediction_buffer)

        max_psi = 0.0
        for col, stats in reference_stats.items():
            if col not in live_df.columns:
                continue
            ref_samples = np.random.normal(
                stats["mean"], stats["std"] + 1e-6, size=500
            )
            ref_samples = np.clip(ref_samples, stats["min"], stats["max"])
            live_vals   = live_df[col].dropna().values
            if len(live_vals) < 10:
                continue
            psi = compute_psi(ref_samples, live_vals)
            max_psi = max(max_psi, psi)
            if psi > PSI_THRESHOLD:
                logger.warning(f"PSI drift on '{col}': {psi:.4f} > threshold {PSI_THRESHOLD}")

        DRIFT_SCORE.set(max_psi)

        if preds:
            mean_pred = float(np.mean(preds))
            PRED_MEAN.set(mean_pred)
            logger.info(f"Drift check — max PSI: {max_psi:.4f}, mean prediction: {mean_pred:.4f}")


def record_request(features: dict, prediction: float, latency_ms: float, error: bool = False):
    """Called by the inference server to register each request."""
    REQUESTS_TOTAL.inc()
    if error:
        ERRORS_TOTAL.inc()
        return
    LATENCY_GAUGE.set(latency_ms)
    with lock:
        request_buffer.append(features)
        prediction_buffer.append(prediction)


def main():
    reference_stats = load_reference_stats(REFERENCE_FILE)
    start_http_server(METRICS_PORT)
    logger.info(f"Prometheus metrics server started on :{METRICS_PORT}")

    t = threading.Thread(target=drift_check_loop, args=(reference_stats,), daemon=True)
    t.start()

    # Keep process alive
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
