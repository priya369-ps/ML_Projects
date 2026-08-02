"""
Feature importance (top predictive words) for the Sentiment Analysis project.

Works for linear models (logistic_regression, linear_svc) which expose
.coef_ -- for naive_bayes, use feature_log_prob_ instead.

Usage:
    python src/feature_importance.py --model model/sentiment_model.pkl --top-n 20
"""

import argparse

import joblib
import numpy as np
import pandas as pd


def analyze(model_path: str, top_n: int = 20):
    bundle = joblib.load(model_path)
    model, vectorizer, model_name = bundle["model"], bundle["vectorizer"], bundle["model_name"]

    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
    elif hasattr(model, "feature_log_prob_"):
        # difference in log-probability between positive and negative class
        coefs = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    else:
        raise ValueError(f"Model type '{model_name}' does not expose interpretable coefficients")

    top_positive_idx = np.argsort(coefs)[-top_n:][::-1]
    top_negative_idx = np.argsort(coefs)[:top_n]

    print(f"\nTop {top_n} words pushing toward POSITIVE sentiment:")
    for idx in top_positive_idx:
        print(f"  {feature_names[idx]:<20} {coefs[idx]:.4f}")

    print(f"\nTop {top_n} words pushing toward NEGATIVE sentiment:")
    for idx in top_negative_idx:
        print(f"  {feature_names[idx]:<20} {coefs[idx]:.4f}")

    result = pd.DataFrame({
        "positive_word": feature_names[top_positive_idx],
        "positive_weight": coefs[top_positive_idx],
        "negative_word": feature_names[top_negative_idx],
        "negative_weight": coefs[top_negative_idx],
    })
    result.to_csv("model/top_words.csv", index=False)
    print("\nSaved to model/top_words.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/sentiment_model.pkl")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    analyze(args.model, args.top_n)
