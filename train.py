from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
TARGET = "churned"
FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
    "payment_method",
    "internet_service",
    "paperless_billing",
    "streaming_bundle",
    "last_login_days",
]
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "last_login_days"]
CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "paperless_billing",
    "streaming_bundle",
]


def make_churn_dataset(rows: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, rows)
    contract = rng.choice(["month_to_month", "one_year", "two_year"], rows, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["card", "bank_transfer", "electronic_check"], rows, p=[0.43, 0.33, 0.24])
    internet = rng.choice(["fiber", "dsl", "none"], rows, p=[0.50, 0.42, 0.08])
    paperless = rng.choice(["yes", "no"], rows, p=[0.68, 0.32])
    streaming = rng.choice(["yes", "no"], rows, p=[0.47, 0.53])

    base_charge = np.where(internet == "fiber", 72, np.where(internet == "dsl", 49, 26))
    monthly_charges = base_charge + rng.normal(0, 9, rows) + np.where(streaming == "yes", 11, 0)
    monthly_charges = np.clip(monthly_charges, 18, 125).round(2)
    support_tickets = rng.poisson(0.7 + (contract == "month_to_month") * 0.45 + (internet == "fiber") * 0.25)
    last_login_days = rng.gamma(shape=2.2, scale=7.0, size=rows).round(0).astype(int)

    logit = (
        -2.15
        + 1.20 * (contract == "month_to_month")
        + 0.75 * (payment == "electronic_check")
        + 0.45 * (internet == "fiber")
        + 0.40 * (paperless == "yes")
        + 0.07 * support_tickets
        + 0.018 * last_login_days
        + 0.012 * (monthly_charges - 60)
        - 0.032 * tenure
    )
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "support_tickets": support_tickets,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            "paperless_billing": paperless,
            "streaming_bundle": streaming,
            "last_login_days": last_login_days,
            TARGET: churned,
        }
    )


def build_pipeline(model) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer churn classifiers.")
    parser.add_argument("--rows", type=int, default=6500)
    parser.add_argument("--test-size", type=float, default=0.22)
    parser.add_argument("--random-state", type=int, default=17)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = make_churn_dataset(args.rows, args.random_state)
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=4, class_weight="balanced", random_state=args.random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=args.random_state),
    }

    results = []
    fitted = {}
    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results.append(evaluate(name, pipeline, X_test, y_test))

    best = max(results, key=lambda row: (row["roc_auc"], row["f1"]))
    for name, pipeline in fitted.items():
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

    with open(MODELS_DIR / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump({"metrics": results, "best_model": best["model"], "features": FEATURES}, handle, indent=2)

    print(pd.DataFrame(results).sort_values("roc_auc", ascending=False).to_string(index=False))
    print(f"\nBest model: {best['model']}")


if __name__ == "__main__":
    main()
