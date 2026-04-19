# Incident Response Playbook
## Credit Scoring MLOps Pipeline

**Scope:** Production SageMaker endpoint `credit-scoring-endpoint` (us-east-1)
**Owner:** ML Engineering
**Last updated:** April 2026

---

## 1. Severity Levels

| Level | Definition | Response time |
|---|---|---|
| **P1 – Critical** | Endpoint down; all requests failing; data exfiltration suspected | 15 minutes |
| **P2 – High** | Error rate > 10%; p95 latency > 5 s; severe prediction drift | 1 hour |
| **P3 – Medium** | Error rate 5–10%; data drift PSI > 0.25; fairness gap detected | 4 hours |
| **P4 – Low** | Minor anomalies; performance degradation < 5%; alert noise | Next business day |

---

## 2. Contacts

| Role | Name | Contact |
|---|---|---|
| On-call ML Engineer | (rotation) | PagerDuty: `mlops-oncall` |
| Platform Lead | | Slack: `#mlops-alerts` |
| Data Privacy Officer | | email: privacy@… |

---

## 3. Incident Runbooks

### INC-001: Endpoint returns 5xx errors

**Trigger:** `HighErrorRate` alert fires (error rate > 5% for 3 min)

1. **Triage** — open the Grafana dashboard `credit-scoring-mlops`; confirm error spike and latency.
2. **Check endpoint health:**
   ```bash
   aws sagemaker describe-endpoint \
     --endpoint-name credit-scoring-endpoint \
     --region us-east-1
   ```
   Look for `EndpointStatus` ≠ `InService`.
3. **Check container logs:**
   ```bash
   aws logs tail /aws/sagemaker/Endpoints/credit-scoring-endpoint \
     --follow --region us-east-1
   ```
4. **If a recent deployment caused it** — rollback to the previous endpoint config:
   ```bash
   aws sagemaker update-endpoint \
     --endpoint-name credit-scoring-endpoint \
     --endpoint-config-name credit-scoring-config-<PREVIOUS_SHA> \
     --region us-east-1
   ```
5. **Post-incident** — open a GitHub issue tagged `incident` with timeline, root cause, and fix.

---

### INC-002: Data drift detected (PSI > 0.10)

**Trigger:** `DataDriftDetected` Prometheus alert

1. Pull the latest drift report:
   ```bash
   python monitoring/drift_detector.py --report-only
   ```
2. Identify which features have the highest PSI scores (see Grafana panel "Max Feature Drift").
3. Verify that the upstream data source has not changed schema or sampling logic.
4. If drift is sustained > 24 h, trigger a pipeline re-run with fresh data:
   ```bash
   dvc repro
   dvc push
   git add dvc.lock metrics/
   git commit -m "retrain: respond to data drift [skip ci]"
   git push origin main
   ```
   This triggers the CD workflow and deploys the new model automatically.
5. Monitor post-deployment drift score — it should fall below 0.10 within 1 h.

---

### INC-003: Prediction drift / concept drift

**Trigger:** `PredictionDrift` alert — rolling mean probability deviated > 5% from 24-hour baseline

1. Compare current vs. baseline class distributions in CloudWatch or Grafana.
2. Check if there was a real-world event (e.g., economic change) that would legitimately shift default rates.
3. If drift appears spurious (model degradation), initiate INC-002 retraining flow.
4. If drift is real (distribution shift), escalate to data science team to evaluate label quality.

---

### INC-004: Fairness violation detected

**Trigger:** `metrics/fairness_report.json` shows AUC gap > 5% between protected groups (SEX, EDUCATION, MARRIAGE)

1. Do not deploy or promote any new model version until resolved.
2. Review `metrics/fairness_report.json`:
   ```bash
   cat metrics/fairness_report.json | python -m json.tool
   ```
3. Investigate whether the gap stems from data imbalance, feature leakage, or label bias.
4. Common fixes: stratified resampling, reweighting, threshold adjustment per group.
5. Re-run `evaluate.py` after fix; verify gap is within acceptable bounds before re-promoting.
6. Log findings in `governance/model_audit_log.md`.

---

### INC-005: CI/CD pipeline failure

**Trigger:** GitHub Actions workflow fails

1. Open the failed workflow in the GitHub Actions UI; expand the failing step.
2. Common causes and fixes:

| Failure | Fix |
|---|---|
| Lint error | Fix flake8 violation in the indicated file |
| Test failure | Run `pytest tests/ -v` locally; fix the broken test |
| DVC push fails | Check AWS credentials in GitHub Secrets; verify S3 bucket permissions |
| ECR push denied | Confirm IAM role has `ecr:PutImage` permission |
| SageMaker timeout | Instance quota may be reached; request limit increase or reduce `INSTANCE_TYPE` |

3. Re-push the fix; the CI run will restart automatically.

---

## 4. Post-Incident Review Template

After every P1/P2 incident, complete this template and attach it to the GitHub issue:

```
## Incident Summary
- Date/time:
- Duration:
- Severity:
- Affected component:

## Timeline
- HH:MM UTC – alert fired
- HH:MM UTC – responder acknowledged
- HH:MM UTC – root cause identified
- HH:MM UTC – fix deployed
- HH:MM UTC – incident resolved

## Root cause
(concise description)

## Impact
(number of failed requests / affected users / business impact)

## Fix applied
(what was changed and why)

## Prevention
(what process or technical change prevents recurrence)
```

---

## 5. Approval Workflow for Model Promotion

All model promotions to production require:

1. Passing CI (lint + tests + sanity train)
2. CD pipeline completes without errors (DVC repro, Docker push, SageMaker deploy)
3. `metrics/fairness_report.json` has no warnings
4. A pull-request review from at least one team member
5. An entry added to `governance/model_audit_log.md` before merging

No model may be deployed directly to `credit-scoring-endpoint` outside this workflow without a documented exception in the audit log.
