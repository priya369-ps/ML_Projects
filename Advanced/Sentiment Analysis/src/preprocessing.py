"""
Data loading and text preprocessing for the Sentiment Analysis project.
"""

import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load the raw sentiment CSV. Expects columns: 'review', 'sentiment'."""
    df = pd.read_csv(path)
    return df


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags/punctuation/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)          # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)        # non-letters
    text = re.sub(r"\s+", " ", text).strip()     # extra whitespace
    return text


def clean_data(df: pd.DataFrame, text_col: str = "review", label_col: str = "sentiment") -> pd.DataFrame:
    """Clean text column and encode the label column to 0/1."""
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)

    if df[label_col].dtype not in ("int64", "int32"):
        df[label_col] = df[label_col].replace({"positive": 1, "negative": 0})
        df[label_col] = pd.to_numeric(df[label_col], errors="raise").astype(int)

    return df


def split_data(df: pd.DataFrame, text_col: str = "review", label_col: str = "sentiment", test_size: float = 0.2):
    """Stratified train/test split."""
    X = df[text_col]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def build_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    """TF-IDF vectorizer with unigrams + bigrams."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
