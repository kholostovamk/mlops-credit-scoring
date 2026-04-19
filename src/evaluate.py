"""
evaluate.py
-----------
Loads the trained model and generates a comprehensive evaluation report:
  - Classification metrics (accuracy, precision, recall, F1, AUC-ROC)
  - Confusion matrix
  - Feature importances
  - Basic fairness audit (performance parity across SEX, EDUCATION, MARRIAGE)

Outputs:
  - metrics/metrics.json
  - metrics/confusion_matrix.csv
  - metrics/feature_importance.csv
  - metrics/fairness_report.json

All metrics are also logged to the parent MLflow run.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STAGED_FILE = os.path.join("data", "staged", "data.csv")
MODEL_PATH = os.path.join("models", "model.joblib")
FEATURE_NAMES_PATH = os.path.join("models", "feature_names.json")
TRAIN_META_PATH = os.path.join("models", "train_meta.json")
METRICS_DIR = "metrics"

RANDOM_STATE = 42
TEST_SIZE = 0.2

FAIRNESS_GROUPS = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {1: "Graduate", 2: "University", 3: "High School"},
    "MARRIAGE": {1: "Married", 2: "Single", 3: "Other"},
}

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")


def load_artifacts():
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = json.load(f)
    with open(TRAIN_META_PATH) as f:
        train_meta = json.load(f)
    return model, feature_names, train_meta


def load_test_data(feature_names):
    df = pd.read_csv(STAGED_FILE)
    X = df[feature_names]
    y = df["DEFAULT"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Also return full df aligned to test indices for fairness analysis
    _, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test, df_test


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def fairness_audit(df_test, y_true, y_pred, y_prob):
    report = {}
    for col, label_map in FAIRNESS_GROUPS.items():
        if col not in df_test.columns:
            continue
        group_results = {}
        for val, name in label_map.items():
            mask = df_test[col].values == val
            if mask.sum() < 30:
                continue
            m = compute_metrics(y_true[mask], y_pred[mask], y_prob[mask])
            group_results[name] = m
        report[col] = group_results

    # Flag large AUC gaps between groups
    warnings = []
    for col, groups in report.items():
        aucs = [v["roc_auc"] for v in groups.values()]
        if aucs and (max(aucs) - min(aucs)) > 0.05:
            warnings.append(
                f"{col}: AUC gap = {max(aucs) - min(aucs):.4f} (threshold 0.05)"
            )
    report["warnings"] = warnings
    if warnings:
        logger.warning(f"Fairness warnings: {warnings}")
    else:
        logger.info("Fairness audit: no significant disparities detected.")
    return report


def extract_feature_importance(model, feature_names):
    try:
        clf = model.named_steps["clf"]
        importances = clf.feature_importances_
        fi = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return fi
    except AttributeError:
        logger.warning("Model does not support feature_importances_.")
        return pd.DataFrame({"feature": feature_names, "importance": [None] * len(feature_names)})


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    model, feature_names, train_meta = load_artifacts()
    X_test, y_test, df_test = load_test_data(feature_names)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Core metrics ──────────────────────────────────────────────────────────
    metrics = compute_metrics(y_test.values, y_pred, y_prob)
    metrics["run_id"] = train_meta.get("run_id", "unknown")
    logger.info(f"Metrics: {metrics}")

    with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_NoDefault", "Actual_Default"],
        columns=["Pred_NoDefault", "Pred_Default"],
    )
    cm_df.to_csv(os.path.join(METRICS_DIR, "confusion_matrix.csv"))
    logger.info(f"Confusion matrix:\n{cm_df}")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = extract_feature_importance(model, feature_names)
    fi.to_csv(os.path.join(METRICS_DIR, "feature_importance.csv"), index=False)
    logger.info(f"Top 5 features:\n{fi.head()}")

    # ── Fairness audit ────────────────────────────────────────────────────────
    fairness = fairness_audit(df_test.reset_index(drop=True),
                              y_test.values, y_pred, y_prob)
    with open(os.path.join(METRICS_DIR, "fairness_report.json"), "w") as f:
        json.dump(fairness, f, indent=2)

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = train_meta.get("run_id")
    if run_id:
        with mlflow.start_run(run_id=run_id):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"eval_{k}", v)
            mlflow.log_artifact(os.path.join(METRICS_DIR, "metrics.json"))
            mlflow.log_artifact(os.path.join(METRICS_DIR, "fairness_report.json"))
            mlflow.log_artifact(os.path.join(METRICS_DIR, "feature_importance.csv"))
        logger.info(f"Metrics logged to MLflow run {run_id}")

    # Print classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
