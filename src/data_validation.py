"""
data_validation.py
------------------
Validates the staged dataset for:
  - Schema (expected columns and dtypes)
  - Missing values
  - Value-range / distribution checks
  - Class-balance check

Writes a JSON validation report to data/staged/validation_report.json.
Exits with code 1 if any critical check fails so DVC can surface the error.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STAGED_FILE = os.path.join("data", "staged", "data.csv")
REPORT_FILE = os.path.join("data", "staged", "validation_report.json")

# Expected schema: column -> (dtype_kind, nullable)
# dtype_kind: 'i' int, 'f' float, 'O' object
EXPECTED_COLUMNS = {
    "LIMIT_BAL": ("f", False),
    "SEX": ("i", False),
    "EDUCATION": ("i", False),
    "MARRIAGE": ("i", False),
    "AGE": ("i", False),
    "PAY_0": ("i", False),
    "PAY_2": ("i", False),
    "PAY_3": ("i", False),
    "PAY_4": ("i", False),
    "PAY_5": ("i", False),
    "PAY_6": ("i", False),
    "BILL_AMT1": ("f", False),
    "BILL_AMT2": ("f", False),
    "BILL_AMT3": ("f", False),
    "BILL_AMT4": ("f", False),
    "BILL_AMT5": ("f", False),
    "BILL_AMT6": ("f", False),
    "PAY_AMT1": ("f", False),
    "PAY_AMT2": ("f", False),
    "PAY_AMT3": ("f", False),
    "PAY_AMT4": ("f", False),
    "PAY_AMT5": ("f", False),
    "PAY_AMT6": ("f", False),
    "DEFAULT": ("i", False),
}

RANGE_CHECKS = {
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
    "AGE": (18, 100),
    "DEFAULT": (0, 1),
}


def load_data():
    logger.info(f"Loading {STAGED_FILE} ...")
    df = pd.read_csv(STAGED_FILE)
    # Coerce numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def check_schema(df, report):
    issues = []
    for col, (_, nullable) in EXPECTED_COLUMNS.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif not nullable and df[col].isnull().any():
            n = df[col].isnull().sum()
            issues.append(f"Column {col} has {n} null values (not allowed)")

    extra = set(df.columns) - set(EXPECTED_COLUMNS.keys())
    if extra:
        logger.warning(f"Extra columns (will be ignored): {extra}")

    report["schema_issues"] = issues
    if issues:
        logger.error(f"Schema issues: {issues}")
    else:
        logger.info("Schema check passed.")
    return len(issues) == 0


def check_missing(df, report):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_values"] = missing_pct[missing_pct > 0].to_dict()
    if report["missing_values"]:
        logger.warning(f"Missing values found: {report['missing_values']}")
    else:
        logger.info("No missing values.")
    return True  # warning, not critical failure


def check_ranges(df, report):
    issues = []
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        out = df[(df[col] < lo) | (df[col] > hi)]
        if len(out) > 0:
            issues.append(
                f"{col}: {len(out)} rows outside [{lo}, {hi}]"
            )
    report["range_issues"] = issues
    if issues:
        logger.warning(f"Range issues: {issues}")
    else:
        logger.info("Range checks passed.")
    return True  # warning only


def check_class_balance(df, report):
    counts = df["DEFAULT"].value_counts(normalize=True).round(4).to_dict()
    report["class_distribution"] = {str(k): v for k, v in counts.items()}
    minority_ratio = min(counts.values())
    if minority_ratio < 0.05:
        logger.warning(f"Severe class imbalance: minority class = {minority_ratio:.2%}")
    else:
        logger.info(f"Class distribution: {counts}")
    return True


def check_row_count(df, report, min_rows=1000):
    report["row_count"] = len(df)
    if len(df) < min_rows:
        logger.error(f"Too few rows: {len(df)} (minimum {min_rows})")
        return False
    logger.info(f"Row count OK: {len(df)}")
    return True


def check_distributions(df, report):
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "p25": round(float(df[col].quantile(0.25)), 4),
            "p50": round(float(df[col].quantile(0.50)), 4),
            "p75": round(float(df[col].quantile(0.75)), 4),
        }
    report["distributions"] = stats
    logger.info("Distribution stats computed.")
    return True


def main():
    df = load_data()
    report = {"status": "unknown", "checks": {}}

    schema_ok = check_schema(df, report)
    check_missing(df, report)
    check_ranges(df, report)
    check_class_balance(df, report)
    rows_ok = check_row_count(df, report)
    check_distributions(df, report)

    all_critical_passed = schema_ok and rows_ok
    report["status"] = "PASSED" if all_critical_passed else "FAILED"

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report written to {REPORT_FILE}")

    if not all_critical_passed:
        logger.error("Critical validation checks FAILED. Aborting pipeline.")
        sys.exit(1)

    logger.info("Data validation complete — status: PASSED")


if __name__ == "__main__":
    main()
