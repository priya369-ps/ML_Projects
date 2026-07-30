"""
train.py
--------
Trains multiple models, compares them, and saves the best one to disk.

Models compared:
- Linear Regression (baseline)
- Random Forest
- XGBoost

Run this file directly to train and save the model:
    python src/train.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from data_loader import load_data, split_data
from preprocessing import build_pipeline
from evaluate import evaluate_model

def get_models():
    """
    Returns a dictionary of models to train and compare.
    """
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    }

def train_and_select_best():
    """
    Trains all models, evaluates them on the test set,
    and saves the best-performing full pipeline (preprocessing + model) to disk.
    """
    #Load and split data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessing = build_pipeline()
    models = get_models()

    best_model_name = None
    best_rmse = np.inf
    best_pipeline = None

    #2. Train and evaluate each model
    for name, model in models.items():
        full_pipeline = Pipeline(steps=[
            ("preprocessing", preprocessing),
            ("model", model)
        ])

        full_pipeline.fit(X_train, y_train)
        y_pred = full_pipeline.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        print(f"Model: {name}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}")

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model_name = name
            best_pipeline = full_pipeline

# 3. Save the best model (path is relative to this script's location,
    #    so it works no matter which folder you run the command from)
    print(f"\nBest model: {best_model_name} (RMSE: {best_rmse:.4f})")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model to {model_path}")


if __name__ == "__main__":
    train_and_select_best()           
