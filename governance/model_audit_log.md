# Model Version Audit Log

This document is the authoritative record of every model version promoted to production for the Credit Scoring MLOps pipeline. It must be updated on every deployment and is versioned alongside the codebase in Git.

---

## How to use this log

When promoting a new model to the `credit-scoring-endpoint` SageMaker endpoint, add an entry below using the template. The CD workflow (`cd.yml`) appends the git SHA and MLflow run ID automatically; the human reviewer fills in the remaining fields before merging the deployment PR.

---

## Log format

| Field | Description |
|---|---|
| **Version** | Monotonically increasing integer |
| **Date (UTC)** | Promotion timestamp |
| **Git SHA** | Commit hash that triggered the CD run |
| **MLflow Run ID** | Experiment run that produced the model |
| **AUC-ROC (test)** | Hold-out test set AUC at time of evaluation |
| **Fairness warnings** | Any SEX / EDUCATION / MARRIAGE AUC gaps > 5% |
| **Approved by** | Name/email of reviewer who merged the deployment PR |
| **Notes** | Dataset version, hyperparameter highlights, known issues |

---

## Entries

### v1 — Baseline

| Field | Value |
|---|---|
| **Version** | 1 |
| **Date (UTC)** | _to be filled on first deployment_ |
| **Git SHA** | _auto-filled by CD_ |
| **MLflow Run ID** | _auto-filled by CD_ |
| **AUC-ROC (test)** | _auto-filled by evaluate.py_ |
| **Fairness warnings** | _auto-filled by evaluate.py_ |
| **Approved by** | _reviewer name_ |
| **Notes** | Initial XGBoost baseline; UCI Default of Credit Card Clients dataset (30k rows, Apr 2026 snapshot). Optuna HPO, 20 trials, 5-fold CV. |

---

_Add new entries above this line, newest first._
