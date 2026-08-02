# Sentiment Analysis

Binary sentiment classification (positive/negative) on text reviews using TF-IDF vectorization and classic ML models, with a Streamlit app for interactive predictions.

## Overview

- Built a text classification pipeline to predict sentiment (positive/negative) from review text.
- Applied TF-IDF vectorization (unigrams + bigrams) for feature extraction.
- Compared Logistic Regression, Naive Bayes, and Linear SVM.
- Extracted top predictive words per class for interpretability.
- Evaluated using accuracy, F1-score, and confusion matrix.

## Dataset

IMDB Movie Reviews dataset (50k labeled reviews, Kaggle) — or substitute any CSV with `review` and `sentiment` columns (`sentiment` as `positive`/`negative` or `1`/`0`). Place the CSV at `data/reviews.csv`.

## Project Structure

```
Sentiment-Analysis/
├── src/
│   ├── preprocessing.py       # text cleaning, TF-IDF vectorizer, split
│   ├── train.py                # trains and saves model/sentiment_model.pkl
│   ├── evaluate.py             # accuracy, F1, confusion matrix
│   └── feature_importance.py   # top predictive words per class
├── app/
│   └── app.py                  # Streamlit app for interactive sentiment prediction
├── model/                      # saved model + outputs (gitignored)
├── data/                       # dataset CSV (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train a model**
```bash
python src/train.py --data data/reviews.csv --model-out model/sentiment_model.pkl --model-name logistic_regression
```

**2. Evaluate it**
```bash
python src/evaluate.py --data data/reviews.csv --model model/sentiment_model.pkl
```

**3. Inspect top predictive words**
```bash
python src/feature_importance.py --model model/sentiment_model.pkl --top-n 20
```

**4. Launch the demo app**
```bash
streamlit run app/app.py
```

## Approach

1. **Preprocessing** (`src/preprocessing.py`) — lowercase, strip HTML tags/punctuation, TF-IDF vectorize with unigrams + bigrams.
2. **Train/test split** — stratified 80/20 split.
3. **Modeling** — Logistic Regression, Naive Bayes, Linear SVM, selectable via `--model-name`.
4. **Evaluation** (`src/evaluate.py`) — accuracy, F1-score, confusion matrix.
5. **Interpretability** (`src/feature_importance.py`) — top words pushing predictions toward positive/negative for linear models.

## Results

| Model               | Accuracy | F1-score |
|---------------------|----------|----------|
| Logistic Regression | TBD      | TBD      |
| Naive Bayes          | TBD      | TBD      |
| Linear SVM           | TBD      | TBD      |

*(Fill in after running `src/evaluate.py`.)*
