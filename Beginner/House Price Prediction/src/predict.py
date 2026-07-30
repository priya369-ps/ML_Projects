"""
predict.py
----------
Loads the saved model and makes predictions on new house data.

Run this file directly to see an example prediction:
    python src/predict.py
"""

import os
import joblib
import pandas as pd


def load_model(path=None):
    """
    Loads the trained pipeline (preprocessing + model) from disk.
    """
    if path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "..", "models", "best_model.pkl")

    return joblib.load(path)


def predict_price(model, input_data: dict):
    """
    Predicts the house price for a single house.

    Args:
        model: Trained pipeline.
        input_data (dict): Feature values, e.g.
            {
                "MedInc": 5.0,
                "HouseAge": 25,
                "AveRooms": 6,
                "AveBedrms": 1,
                "Population": 1000,
                "AveOccup": 3,
                "Latitude": 34.0,
                "Longitude": -118.0
            }

    Returns:
        float: Predicted price (in $100,000s, matching the original dataset units).
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Make prediction
    predicted_price = model.predict(df)[0]

    return predicted_price


if __name__ == "__main__":
    # Load trained model
    model = load_model()

    # Example house
    sample_house = {
        "MedInc": 5.0,
        "HouseAge": 25,
        "AveRooms": 6,
        "AveBedrms": 1,
        "Population": 1000,
        "AveOccup": 3,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }

    # Predict price
    price = predict_price(model, sample_house)

    # Convert to dollars
    print(f"Predicted house price: ${price * 100000:,.2f}")