# Credit Scoring MLOps Pipeline

End-to-end production MLOps pipeline for predicting credit card payment default, built on DVC, MLflow, Docker, AWS SageMaker, and GitHub Actions.

**Dataset:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (UCI, 30 000 rows, 23 features)
**Model:** XGBoost classifier, tuned with Optuna (20 HPO trials, 5-fold CV)
**Bonus components:** MLflow experiment tracking & model registry + Governance & incident response

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  GitHub repository                                                  │
│  ┌─────────────┐  push / PR   ┌──────────────────────────────────┐ │
│  │  Developer  │─────────────▶│  GitHub Actions CI               │ │
│  └─────────────┘              │  lint → tests → sanity train     │ │
│                               └──────────────┬───────────────────┘ │
│                                              │ merge to main        │
│                               ┌──────────────▼───────────────────┐ │
│                               │  GitHub Actions CD               │ │
│                               │  dvc repro → docker build/push   │ │
│                               │  → SageMaker endpoint update     │ │
│                               └──────────────┬───────────────────┘ │
└──────────────────────────────────────────────┼─────────────────────┘
                                               │
         ┌─────────────────┬──────────────────┼──────────────────┐
         ▼                 ▼                  ▼                  ▼
    S3 (DVC remote)   ECR (image)     SageMaker endpoint    MLflow registry
    dvc-store/        mlops-credit-   credit-scoring-       credit-scoring-
                      scoring:sha     endpoint              xgb (v1, v2 …)

                                               │ invocations
                                    ┌──────────▼──────────┐
                                    │  Prometheus +        │
                                    │  Grafana dashboard   │
                                    │  (drift, latency,    │
                                    │   error rate)        │
                                    └─────────────────────┘
```

---

## Repository layout

```
.
├─ .dvc/config                  DVC remote configuration (S3)
├─ .github/workflows/
│  ├─ ci.yml                    CI: lint, tests, sanity train
│  └─ cd.yml                    CD: DVC repro, Docker, SageMaker deploy
├─ data/
│  ├─ raw/                      Raw XLS (DVC-tracked)
│  └─ staged/                   Cleaned CSV + validation report (DVC-tracked)
├─ governance/
│  ├─ model_audit_log.md        Version audit trail
│  └─ incident_playbook.md      Runbooks for P1–P4 incidents
├─ inference/
│  ├─ predict.py                SageMaker inference hooks (model_fn, input_fn, predict_fn, output_fn)
│  └─ serve.py                  Flask server exposing /ping and /invocations
├─ metrics/                     JSON metrics + fairness report (DVC-tracked)
├─ mlruns/                      Local MLflow tracking (gitignored)
├─ models/                      Trained model artefacts (DVC-tracked)
├─ monitoring/
│  ├─ alert_rules.yml           Prometheus alerting rules
│  ├─ drift_detector.py         PSI-based drift sidecar
│  ├─ prometheus.yml            Prometheus scrape config
│  └─ grafana/dashboard.json    Grafana dashboard definition
├─ src/
│  ├─ data_ingest.py
│  ├─ data_validation.py
│  ├─ train_and_tune.py
│  └─ evaluate.py
├─ tests/
│  └─ test_pipeline.py
├─ Dockerfile
├─ dvc.yaml
├─ params.yaml
└─ requirements.txt
```

---

## Local setup

### Prerequisites

- Python 3.11
- Docker Desktop (running)
- AWS CLI configured (`aws configure`) with SageMaker, ECR, and S3 access
- Git

### 1. Clone and install

```bash
git clone https://github.com/kholostovamk/mlops-credit-scoring.git
cd mlops-credit-scoring
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure DVC remote

Create the S3 bucket first (one-time):

```bash
aws s3 mb s3://mlops-credit-scoring-dvc --region us-east-1
```

The `.dvc/config` already points to this bucket. Credentials are picked up from your AWS CLI config automatically.

### 3. Run the full pipeline

```bash
dvc repro        # runs ingest → validate → train → evaluate
dvc push         # push data + model artefacts to S3
```

After the pipeline completes:

```bash
cat metrics/metrics.json          # AUC-ROC, F1, accuracy, etc.
cat metrics/fairness_report.json  # per-group performance breakdown
```

### 4. View MLflow experiments

```bash
mlflow ui --backend-store-uri mlruns
# open http://localhost:5000
```

---

## Docker build and local inference test

```bash
# Build
docker build -t credit-scoring:local .

# Run the inference server locally (model is baked into the image)
docker run -p 8080:8080 \
  -e SM_MODEL_DIR=/opt/program/models \
  credit-scoring:local

# Smoke-test with curl
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
      "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
      "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
      "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
      "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
      "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000
    }]
  }'
```

---

## AWS SageMaker deployment (manual first-time setup)

The CD workflow handles this automatically on every merge to `main`. For the initial manual deploy:

### Step 1 — Create ECR repository

```bash
aws ecr create-repository \
  --repository-name mlops-credit-scoring \
  --region us-east-1
```

### Step 2 — Push Docker image to ECR

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ECR

docker tag credit-scoring:local $ECR/mlops-credit-scoring:latest
docker push $ECR/mlops-credit-scoring:latest
```

### Step 3 — Create SageMaker endpoint

> **Note:** The model is baked into the Docker image (`ENV SM_MODEL_DIR /opt/program/models`), so no S3 model upload is needed.

```python
import boto3

sm    = boto3.client("sagemaker", region_name="us-east-1")
ROLE  = "<your-sagemaker-execution-role-arn>"
IMAGE = f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/mlops-credit-scoring:latest"

sm.create_model(
    ModelName="credit-scoring-model-v1",
    PrimaryContainer={
        "Image": IMAGE,
        "Mode": "SingleModel",
        "Environment": {"SM_MODEL_DIR": "/opt/program/models"},
    },
    ExecutionRoleArn=ROLE,
)
sm.create_endpoint_config(
    EndpointConfigName="credit-scoring-config-v1",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": "credit-scoring-model-v1",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large",
    }],
)
sm.create_endpoint(
    EndpointName="credit-scoring-endpoint",
    EndpointConfigName="credit-scoring-config-v1",
)
print("Endpoint creation started — takes ~5 minutes to be InService.")
```

### Step 4 — Test the live endpoint

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
response = runtime.invoke_endpoint(
    EndpointName="credit-scoring-endpoint",
    ContentType="application/json",
    Body=json.dumps({"instances": [{
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
        "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
        "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
        "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
    }]}),
)
print(json.loads(response["Body"].read()))
```

---

## CI/CD setup (GitHub Secrets)

Add the following secrets to your GitHub repository under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN (e.g. `arn:aws:iam::…:role/SageMakerRole`) |

**CI triggers:** any push to a non-`main` branch, and all pull requests to `main`.
**CD triggers:** every merge / push to `main`.

---

## Monitoring

Start the full local monitoring stack (inference server + Prometheus + Grafana + drift detector):

```bash
docker compose up --build
# Inference server: http://localhost:8080/ping
# Prometheus:       http://localhost:9090
# Grafana:          http://localhost:3000  (admin / admin)
```

The Grafana dashboard (`monitoring/grafana/dashboard.json`) is provisioned automatically on startup — no manual import needed. It shows request throughput, p95 latency, error rate, and PSI drift score.

Alert rules in `monitoring/alert_rules.yml` fire when:
- Inference error rate > 5% for 2 minutes
- p95 latency > 500 ms for 5 minutes
- PSI drift score > 0.2 (significant distribution shift)

---

## Running tests

```bash
pytest tests/ -v --cov=src --cov=inference --cov-report=term-missing
```

---

## Governance

Model promotion requires a passing CI run, no fairness warnings, and a peer-reviewed PR. Every production deployment is logged in [`governance/model_audit_log.md`](governance/model_audit_log.md). Incident response procedures are documented in [`governance/incident_playbook.md`](governance/incident_playbook.md).

---

## Key design decisions

See the Final Report PDF for the full discussion. Brief summary:

- **XGBoost over deep learning** — better performance on tabular data at this scale, faster CI sanity trains, interpretable feature importances.
- **Optuna over GridSearch** — efficient Bayesian HPO; 20 trials explore the space faster than exhaustive search.
- **DVC on S3** — decouples large artefacts from Git while keeping reproducibility; `dvc repro` guarantees identical results from any machine.
- **Two-stage Docker build** — separates heavy compilation from the lean runtime image, reducing final image size by ~60%.
- **PSI for drift detection** — statistically principled, no labelled data required at inference time.
