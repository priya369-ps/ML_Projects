"""
Streamlit app for the Sentiment Analysis project.

Run with:
    streamlit run app/app.py
"""

import os
import sys

import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import clean_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "sentiment_model.pkl")

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("Sentiment Analysis")
st.write("Enter a review or piece of text to predict whether it's positive or negative.")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

if bundle is None:
    st.warning(
        "No trained model found at `model/sentiment_model.pkl`. "
        "Run `python src/train.py --data data/reviews.csv` first."
    )
else:
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    text_input = st.text_area("Enter text", height=150, placeholder="Type or paste a review here...")

    if st.button("Predict Sentiment"):
        if not text_input.strip():
            st.error("Please enter some text.")
        else:
            cleaned = clean_text(text_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            label = "Positive 🙂" if pred == 1 else "Negative 🙁"

            st.subheader(label)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                st.metric("Confidence", f"{max(proba):.1%}")
