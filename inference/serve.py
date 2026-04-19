"""
serve.py – Flask inference server for SageMaker custom container.

SageMaker sends:
  GET  /ping         → health check (must return 200)
  POST /invocations  → inference request
"""

import os
import logging
from flask import Flask, request, Response
from predict import model_fn, input_fn, predict_fn, output_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/program/models")
logger.info(f"Loading model from {MODEL_DIR}")
MODEL = model_fn(MODEL_DIR)
logger.info("Model loaded successfully.")


@app.route("/ping", methods=["GET"])
def ping():
    """SageMaker health check."""
    return Response("OK", status=200, mimetype="text/plain")


@app.route("/invocations", methods=["POST"])
def invoke():
    """SageMaker inference endpoint."""
    content_type = request.content_type or "application/json"
    accept = request.headers.get("Accept", "application/json")

    try:
        data = input_fn(request.get_data(as_text=True), content_type)
        prediction = predict_fn(data, MODEL)
        body, mime = output_fn(prediction, accept)
        return Response(body, status=200, mimetype=mime)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return Response(str(e), status=500, mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
