# Customer churn lab

Practical churn prediction project for a subscription business. The dataset is generated locally with stable rules and noise, so the project is reproducible without downloading customer data.

## What is inside

- Synthetic customer account generator with realistic churn drivers.
- Three sklearn baselines: logistic regression, random forest and gradient boosting.
- A consistent preprocessing pipeline for numeric and categorical features.
- CLI inference script that loads a trained artifact from `models/`.
- Notebook for visual EDA with Matplotlib and Seaborn.

## Dataset

`train.py` builds a synthetic customer dataset with these fields:

| Feature | Meaning |
| --- | --- |
| `tenure_months` | Customer age in months |
| `monthly_charges` | Current recurring charge |
| `support_tickets` | Recent support ticket count |
| `contract_type` | `month_to_month`, `one_year`, `two_year` |
| `payment_method` | Billing method |
| `internet_service` | `fiber`, `dsl`, `none` |
| `paperless_billing` | Account billing preference |
| `streaming_bundle` | Add-on indicator |
| `last_login_days` | Days since last account login |

Target: `churned` (`1` means the customer churned).

## Model comparison

Running `python train.py` writes exact metrics to `models/comparison.json`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | generated at train time | generated at train time | generated at train time | generated at train time | generated at train time |
| Random forest | generated at train time | generated at train time | generated at train time | generated at train time | generated at train time |
| Gradient boosting | generated at train time | generated at train time | generated at train time | generated at train time | generated at train time |

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

Example prediction:

```bash
python predict.py \
  --model boosting \
  --tenure-months 8 \
  --monthly-charges 91.4 \
  --support-tickets 3 \
  --contract-type month_to_month \
  --payment-method electronic_check \
  --internet-service fiber \
  --paperless-billing yes \
  --streaming-bundle yes \
  --last-login-days 19
```

## Notes

The project keeps data generation in code so experiments are repeatable and no private customer data is checked into the repository. Artifacts under `models/*.joblib` are intentionally ignored and can be regenerated.
