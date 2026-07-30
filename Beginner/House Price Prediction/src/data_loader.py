"""
data_loader.py
--------------
Loads the House Price dataset and splits it into train/test sets.

NOTE: This project generates a synthetic housing dataset (fixed random seed)
instead of downloading one. This keeps the project fully offline and
reproducible. To use a REAL dataset instead (e.g. Kaggle's "House Prices -
Advanced Regression Techniques"), just replace load_data() below to read
your CSV with pd.read_csv(), keeping the same column names.

Each row = one house. Target = 'PRICE' (in $100,000s).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(n_samples=2000, random_state=42):
    """
    Generates a synthetic but realistic housing dataset.

    Features mimic the well-known California Housing dataset schema:
        MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
        Latitude, Longitude

    Returns:
        df (DataFrame): features + target column 'PRICE'
    """
    rng = np.random.default_rng(random_state)

    med_inc = rng.uniform(1, 15, n_samples)
    house_age = rng.uniform(1, 52, n_samples)
    ave_rooms = rng.uniform(2, 10, n_samples)
    ave_bedrms = ave_rooms * rng.uniform(0.15, 0.35, n_samples)
    population = rng.uniform(3, 5000, n_samples)
    ave_occup = rng.uniform(1, 6, n_samples)
    latitude = rng.uniform(32, 42, n_samples)
    longitude = rng.uniform(-124, -114, n_samples)

    # True underlying relationship + noise, so models have something real to learn
    price = (
        3.0 * med_inc
        - 0.02 * house_age
        + 0.5 * ave_rooms
        - 1.0 * ave_bedrms
        - 0.0005 * population
        -0.3 * ave_occup
        + rng.normal(0, 1.5, n_samples)
    )
    price = np.clip(price, 0.5, None)  # no negative prices  

    df = pd.DataFrame({
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
        "PRICE": price
    })

    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into train and test sets.

    Args:
        df (DataFrame): full dataset
        test_size (float): fraction of data to use for testing
        random_state (int): for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["PRICE"])
    y = df["PRICE"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    # Quick check: run this file directly to see the data shape
    df = load_data()
    print("Dataset shape:", df.shape)
    print(df.head()) 