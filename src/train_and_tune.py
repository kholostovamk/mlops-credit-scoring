"""
train_and_tune.py
-----------------
Trains an XGBoost classifier on the staged credit-default dataset with
hyperparameter optimisation via Optuna.  All runs are tracked with MLflow.

Outputs:
  - models/model.joblib  (best model)
  - models/feature_names.json
"""

import os
import json
import logging
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STAGED_FILE = os.path.join("data", "staged", "data.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

RANDOM_STATE = 42
N_TRIALS = 20          # Optuna HPO trials
CV_FOLDS = 5
TEST_SIZE = 0.2

MLFLOW_EXPERIMENT = "credit-scoring"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")


def load_data():
    logger.info(f"Loading {STAGED_FILE}")
    df = pd.read_csv(STAGED_FILE)
    X = df.drop(columns=["DEFAULT"])
    y = df["DEFAULT"]
    return X, y


def build_pipeline(params: dict) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def objective(trial, X_train, y_train):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    pipeline = build_pipeline(params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return scores.mean()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y = load_data()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ── Hyperparameter Optimisation ────────────────────────────────────────────
    logger.info(f"Starting Optuna HPO with {N_TRIALS} trials ...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="xgb-credit")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    best_params = study.best_params
    logger.info(f"Best CV AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    # ── Final training with best params ───────────────────────────────────────
    with mlflow.start_run(run_name="best-xgb") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", len(feature_names))
        mlflow.log_metric("best_cv_auc", study.best_value)

        best_pipeline = build_pipeline(best_params)
        best_pipeline.fit(X_train, y_train)

        # Evaluate on hold-out test set
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("test_auc", test_auc)
        logger.info(f"Test AUC: {test_auc:.4f}")

        # Log model to MLflow registry
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            registered_model_name="credit-scoring-xgb",
        )

        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

    # Save model and feature list locally for DVC
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f)
    logger.info(f"Feature names saved to {FEATURE_NAMES_PATH}")

    # Save run metadata for evaluate.py
    meta = {"run_id": run_id, "test_auc": test_auc, "best_params": best_params}
    with open(os.path.join(MODEL_DIR, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
