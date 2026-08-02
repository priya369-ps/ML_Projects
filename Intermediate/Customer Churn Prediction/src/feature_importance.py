"""
Feature importance analysis (SHAP) for the Customer Churn Prediction project.

Usage:
    python src/feature_importance.py --data data/telco_churn.csv --model model/churn_model.pkl
"""

import argparse

import joblib
import matplotlib.pyplot as plt
import shap

from preprocessing import load_data, clean_data, split_data, get_feature_names


def analyze(data_path: str, model_path: str):
    bundle = joblib.load(model_path)
    model, preprocessor = bundle["model"], bundle["preprocessor"]

    df = clean_data(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df)
    X_test_proc = preprocessor.transform(X_test)

    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    feature_names = get_feature_names(preprocessor, numeric_cols, categorical_cols)

    if bundle["model_name"] not in ("random_forest", "xgboost"):
        raise ValueError("SHAP TreeExplainer requires a tree-based model (random_forest or xgboost)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_proc)

    shap.summary_plot(shap_values, X_test_proc, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("model/shap_summary.png")
    print("Saved SHAP summary plot to model/shap_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="model/churn_model.pkl")
    args = parser.parse_args()
    analyze(args.data, args.model)
