"""
Unit tests for the MLOps credit scoring pipeline.
Run with: pytest tests/ -v
"""

import json
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

# Make src importable without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "DEFAULT",
]


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 300
    df = pd.DataFrame(np.random.randn(n, len(FEATURE_COLS) - 1), columns=FEATURE_COLS[:-1])
    df["LIMIT_BAL"] = np.abs(df["LIMIT_BAL"]) * 50000
    df["SEX"] = np.random.choice([1, 2], n)
    df["EDUCATION"] = np.random.choice([1, 2, 3], n)
    df["MARRIAGE"] = np.random.choice([1, 2, 3], n)
    df["AGE"] = np.random.randint(22, 65, n)
    df["DEFAULT"] = np.random.choice([0, 1], n, p=[0.78, 0.22])
    return df


@pytest.fixture
def staged_csv(sample_df, tmp_path):
    p = tmp_path / "data" / "staged"
    p.mkdir(parents=True)
    csv_path = p / "data.csv"
    sample_df.to_csv(csv_path, index=False)
    return str(csv_path)


# ─── Data validation tests ────────────────────────────────────────────────────

class TestDataValidation:
    def test_all_expected_columns_present(self, sample_df):
        expected = [c for c in FEATURE_COLS]
        for col in expected:
            assert col in sample_df.columns

    def test_no_missing_values_in_sample(self, sample_df):
        assert sample_df.isnull().sum().sum() == 0

    def test_default_binary(self, sample_df):
        assert set(sample_df["DEFAULT"].unique()).issubset({0, 1})

    def test_sex_valid_range(self, sample_df):
        assert sample_df["SEX"].between(1, 2).all()

    def test_age_valid_range(self, sample_df):
        assert sample_df["AGE"].between(18, 100).all()

    def test_row_count(self, sample_df):
        assert len(sample_df) >= 100


# ─── Inference script tests ───────────────────────────────────────────────────

class TestInference:
    def test_input_fn_json(self):
        from inference.predict import input_fn
        payload = json.dumps({"instances": [{
            col: 0.0 for col in FEATURE_COLS if col != "DEFAULT"
        }]})
        df = input_fn(payload, "application/json")
        assert len(df) == 1

    def test_input_fn_list(self):
        from inference.predict import input_fn
        payload = json.dumps([{col: 0.0 for col in FEATURE_COLS if col != "DEFAULT"}])
        df = input_fn(payload, "application/json")
        assert len(df) == 1

    def test_input_fn_unsupported_type(self):
        from inference.predict import input_fn
        with pytest.raises(ValueError):
            input_fn("data", "application/xml")

    def test_output_fn_json(self):
        from inference.predict import output_fn
        pred = {"probabilities": [0.3, 0.7], "predictions": [0, 1]}
        body, mime = output_fn(pred, "application/json")
        assert mime == "application/json"
        parsed = json.loads(body)
        assert "predictions" in parsed

    def test_output_fn_csv(self):
        from inference.predict import output_fn
        pred = {"probabilities": [0.3], "predictions": [0]}
        body, mime = output_fn(pred, "text/csv")
        assert mime == "text/csv"
        assert "probability" in body

    def test_predict_fn_with_real_model(self, sample_df, tmp_path):
        """Train a tiny model and verify predict_fn returns expected structure."""
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        from inference.predict import predict_fn

        feature_names = [c for c in FEATURE_COLS if c != "DEFAULT"]
        X = sample_df[feature_names]
        y = sample_df["DEFAULT"]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=5, random_state=0,
                                  use_label_encoder=False, eval_metric="logloss")),
        ])
        pipe.fit(X, y)

        model_path = tmp_path / "model.joblib"
        feat_path = tmp_path / "feature_names.json"
        joblib.dump(pipe, model_path)
        feat_path.write_text(json.dumps(feature_names))

        artifacts = {"model": pipe, "feature_names": feature_names}
        result = predict_fn(X.head(5), artifacts)

        assert "probabilities" in result
        assert "predictions" in result
        assert len(result["probabilities"]) == 5
        assert all(p in [0, 1] for p in result["predictions"])


# ─── Metrics structure tests ──────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_keys(self, tmp_path):
        """Verify evaluate.compute_metrics returns all required keys."""
        from src.evaluate import compute_metrics
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0
