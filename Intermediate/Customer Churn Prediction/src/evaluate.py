"""
Evaluation for the Customer Churn Prediction project.

Usage:
    python src/evaluate.py --data data/telco_churn.csv --model model/churn_model.pkl
"""

import argparse

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from preprocessing import load_data, clean_data, split_data


def evaluate(data_path: str, model_path: str, plot: bool = True):
    bundle = joblib.load(model_path)
    model, preprocessor = bundle["model"], bundle["preprocessor"]

    df = clean_data(load_data(data_path))
    _, X_test, _, y_test = split_data(df)
    X_test_proc = preprocessor.transform(X_test)

    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-score: {f1:.4f}")

    if plot:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{bundle['model_name']} (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("model/roc_curve.png")
        print("Saved ROC curve to model/roc_curve.png")

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(cmap="Blues")
        plt.title(f"Confusion Matrix -- {bundle['model_name']}")
        plt.tight_layout()
        plt.savefig("model/confusion_matrix.png")
        print("Saved confusion matrix to model/confusion_matrix.png")

    return {"roc_auc": auc, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="model/churn_model.pkl")
    args = parser.parse_args()
    evaluate(args.data, args.model)
