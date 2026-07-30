"""
evaluate.py
-----------
Simple evaluation metrics for regression models.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Computes standard regression metrics.

    Args:
        y_true: actual values
        y_pred: predicted values

    Returns:
        dict with RMSE, MAE, R2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}