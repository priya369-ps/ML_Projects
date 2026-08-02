"""
Model training for the Sentiment Analysis project.

Usage:
    python src/train.py --data data/reviews.csv --model-out model/sentiment_model.pkl
"""

import argparse
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import load_data, clean_data, split_data, build_vectorizer

RANDOM_STATE = 42


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "naive_bayes": MultinomialNB(),
        "linear_svc": LinearSVC(random_state=RANDOM_STATE),
    }


def train(data_path: str, model_out: str, model_name: str = "logistic_regression"):
    df = clean_data(load_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df)

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = get_models()[model_name]
    model.fit(X_train_vec, y_train)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "model_name": model_name},
        model_out,
    )
    print(f"Saved trained model + vectorizer to {model_out}")

    return model, vectorizer, X_test, y_test, X_test_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the raw CSV")
    parser.add_argument("--model-out", default="model/sentiment_model.pkl")
    parser.add_argument(
        "--model-name",
        default="logistic_regression",
        choices=["logistic_regression", "naive_bayes", "linear_svc"],
    )
    args = parser.parse_args()
    train(args.data, args.model_out, args.model_name)
