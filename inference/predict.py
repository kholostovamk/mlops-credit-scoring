"""
predict.py  –  SageMaker inference script
------------------------------------------
SageMaker calls four hooks during inference:
  model_fn     – load model from /opt/ml/model
  input_fn     – deserialise incoming request
  predict_fn   – run inference
  output_fn    – serialise response

Supported content types: application/json, text/csv
"""

import os
import io
import json
import logging
import time
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# SageMaker stores the model artefact here after unpacking model.tar.gz
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names.json")


def model_fn(model_dir: str):
    """Load model and feature list from model_dir."""
    model_path = os.path.join(model_dir, "model.joblib")
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    feature_names_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_names_path) as f:
        feature_names = json.load(f)

    return {"model": model, "feature_names": feature_names}


def input_fn(request_body, content_type: str = "application/json"):
    """Deserialise request body into a pandas DataFrame."""
    logger.info(f"Content-Type: {content_type}, Body length: {len(request_body)}")

    if content_type == "application/json":
        data = json.loads(request_body)
        # Accept {"instances": [...]} or raw list or single dict
        if isinstance(data, dict) and "instances" in data:
            data = data["instances"]
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)

    elif content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        # Caller must send columns in the correct order (no header row)

    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    return df


def predict_fn(input_data: pd.DataFrame, model_artifacts: dict):
    """Run inference and return probability + binary prediction."""
    model = model_artifacts["model"]
    feature_names = model_artifacts["feature_names"]

    # Align columns if present, otherwise assume correct order
    if set(feature_names).issubset(set(input_data.columns)):
        input_data = input_data[feature_names]
    else:
        input_data.columns = feature_names

    start = time.time()
    probabilities = model.predict_proba(input_data)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    latency_ms = (time.time() - start) * 1000
    logger.info(f"Inference latency: {latency_ms:.1f} ms for {len(input_data)} rows")

    return {"probabilities": probabilities.tolist(), "predictions": predictions.tolist()}


def output_fn(prediction, accept: str = "application/json"):
    """Serialise the prediction dict to the requested format."""
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    elif accept == "text/csv":
        rows = zip(prediction["probabilities"], prediction["predictions"])
        csv = "probability,prediction\n" + "\n".join(
            f"{p:.6f},{int(l)}" for p, l in rows
        )
        return csv, "text/csv"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
