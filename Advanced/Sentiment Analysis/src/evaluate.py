"""
Evaluation for the Sentiment Analysis project.

Usage:
    python src/evaluate.py --data data/reviews.csv --model model/sentiment_model.pkl
"""

import argparse

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from preprocessing import load_data, clean_data, split_data


def evaluate(data_path: str, model_path: str, plot: bool = True):
    bundle = joblib.load(model_path)
    model, vectorizer = bundle["model"], bundle["vectorizer"]

    df = clean_data(load_data(data_path))
    _, X_test, _, y_test = split_data(df)
    X_test_vec = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")

    if plot:
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(cmap="Blues")
        plt.title(f"Confusion Matrix -- {bundle['model_name']}")
        plt.tight_layout()
        plt.savefig("model/confusion_matrix.png")
        print("Saved confusion matrix to model/confusion_matrix.png")

    return {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="model/sentiment_model.pkl")
    args = parser.parse_args()
    evaluate(args.data, args.model)
