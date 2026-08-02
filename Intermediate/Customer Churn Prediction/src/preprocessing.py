"""
Data loading and preprocessing for the Customer Churn Prediction project.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: fix TotalCharges, drop ID column, encode target."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    df["Churn"] = pd.to_numeric(df["Churn"], errors="raise").astype(int)

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric scaling + categorical encoding."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols, categorical_cols) -> list:
    """Recover human-readable feature names after transformation."""
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    )
    return numeric_cols + cat_names


def split_data(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2):
    """Stratified train/test split. Do this BEFORE any resampling (e.g. SMOTE)."""
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test