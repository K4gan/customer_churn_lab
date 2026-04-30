from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_NAMES = {"logistic": "logistic_regression.joblib", "forest": "random_forest.joblib", "boosting": "gradient_boosting.joblib"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn from account attributes.")
    parser.add_argument("--model", choices=MODEL_NAMES.keys(), default="boosting")
    parser.add_argument("--tenure-months", type=int, required=True)
    parser.add_argument("--monthly-charges", type=float, required=True)
    parser.add_argument("--support-tickets", type=int, required=True)
    parser.add_argument("--contract-type", choices=["month_to_month", "one_year", "two_year"], required=True)
    parser.add_argument("--payment-method", choices=["card", "bank_transfer", "electronic_check"], required=True)
    parser.add_argument("--internet-service", choices=["fiber", "dsl", "none"], required=True)
    parser.add_argument("--paperless-billing", choices=["yes", "no"], required=True)
    parser.add_argument("--streaming-bundle", choices=["yes", "no"], required=True)
    parser.add_argument("--last-login-days", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = MODELS_DIR / MODEL_NAMES[args.model]
    if not model_path.exists():
        raise SystemExit(f"Model artifact not found: {model_path}. Run python train.py first.")

    row = pd.DataFrame(
        [
            {
                "tenure_months": args.tenure_months,
                "monthly_charges": args.monthly_charges,
                "support_tickets": args.support_tickets,
                "contract_type": args.contract_type,
                "payment_method": args.payment_method,
                "internet_service": args.internet_service,
                "paperless_billing": args.paperless_billing,
                "streaming_bundle": args.streaming_bundle,
                "last_login_days": args.last_login_days,
            }
        ]
    )
    pipeline = joblib.load(model_path)
    probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    print(f"prediction={prediction}")
    print(f"churn_probability={probability:.4f}")


if __name__ == "__main__":
    main()
