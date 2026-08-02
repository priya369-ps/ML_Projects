"""
Model training for the Customer Churn Prediction project.

Usage:
    python src/train.py --data data/telco_churn.csv --model-out model/churn_model.pkl
"""

import argparse
import os

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb

from preprocessing import load_data, clean_data, build_preprocessor, split_data

RANDOM_STATE = 42


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def train(data_path: str, model_out: str, model_name: str = "xgboost"):
    df = clean_data(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # SMOTE applied only to the training set, after the split
    print(f"Before SMOTE: {np.bincount(y_train)}")
    X_train_sm, y_train_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train_proc, y_train
    )
    print(f"After SMOTE: {np.bincount(y_train_sm)}")

    model = get_models()[model_name]
    model.fit(X_train_sm, y_train_sm)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(
        {"model": model, "preprocessor": preprocessor, "model_name": model_name},
        model_out,
    )
    print(f"Saved trained model + preprocessor to {model_out}")

    return model, preprocessor, X_test, y_test, X_test_proc, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the raw CSV")
    parser.add_argument("--model-out", default="model/churn_model.pkl")
    parser.add_argument(
        "--model-name",
        default="xgboost",
        choices=["logistic_regression", "random_forest", "xgboost"],
    )
    args = parser.parse_args()
    train(args.data, args.model_out, args.model_name)
