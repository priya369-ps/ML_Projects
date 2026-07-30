"""
preprocessing.py
----------------
Feature engineering + preprocessing pipeline.

We keep this simple:
1. Add a couple of engineered features.
2. Scale all numeric features using StandardScaler.
Everything is wrapped in a single sklearn Pipeline so the SAME steps
run identically during training and prediction (no data leakage).
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def add_features(X):
    """
    Adds new engineered features to the dataset.

    - RoomsPerHousehold: average rooms per household
    - BedroomsPerRoom: ratio of bedrooms to total rooms
    - PopulationPerHousehold: average people per household

    Args:
        X (DataFrame): input features

    Returns:
        DataFrame with extra columns added
    """
    X = X.copy()
    X["RoomsPerHousehold"] = X["AveRooms"] / X["AveOccup"]
    X["BedroomsPerRoom"] = X["AveBedrms"] / X["AveRooms"]
    X["PopulationPerHousehold"] = X["Population"] / X["AveOccup"]
    return X

def build_pipeline():
    """
    Constructs the preprocessing pipeline.

    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(add_features)),
        ("scaler", StandardScaler())
    ])
    return pipeline