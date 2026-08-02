# ML Projects

This repository collects multiple machine learning projects at different experience levels. Each project is self-contained with its own data, model files, training and evaluation scripts, and user-facing application code when available.

## Root folder structure

```
ML_Projects/
├── Advanced/
│   └── Sentiment Analysis/
│       ├── README.md
│       ├── requirements.txt
│       ├── app/
│       │   └── app.py
│       ├── data/
│       │   └── reviews.csv
│       ├── model/
│       │   └── top_words.csv
│       └── src/
│           ├── evaluate.py
│           ├── feature_importance.py
│           ├── preprocessing.py
│           └── train.py
├── Beginner/
│   └── House Price Prediction/
│       ├── readme.md
│       ├── requirements.txt
│       ├── app/
│       │   └── app.py
│       ├── models/
│       └── src/
│           ├── __init__.py
│           ├── data_loader.py
│           ├── evaluate.py
│           ├── predict.py
│           ├── preprocessing.py
│           └── train.py
└── Intermediate/
    └── Customer Churn Prediction/
        ├── README.md
        ├── requirements.txt
        ├── app/
        │   └── app.py
        ├── data/
        │   └── telco_churn.csv
        ├── model/
        └── src/
            ├── evaluate.py
            ├── feature_importance.py
            ├── preprocessing.py
            └── train.py
```

## Projects overview

### Advanced: Sentiment Analysis
A text classification project using movie review data. It includes text preprocessing, feature extraction, model training, evaluation, and a Streamlit demo app.

- `data/reviews.csv`: labeled movie reviews dataset.
- `src/preprocessing.py`: text cleaning and vectorization.
- `src/train.py`: train a classifier and save model artifacts.
- `src/evaluate.py`: compute evaluation metrics.
- `app/app.py`: demo interface for entering review text and viewing sentiment predictions.

### Beginner: House Price Prediction
A regression project that predicts housing prices from structured features. It demonstrates data loading, preprocessing, training, and model prediction.

- `src/data_loader.py`: loads and prepares the house price dataset.
- `src/preprocessing.py`: scales and transforms numeric features.
- `src/train.py`: trains a regression model.
- `src/evaluate.py`: evaluates the model performance.
- `src/predict.py`: generates price predictions for new examples.
- `app/app.py`: optional app interface for evaluating price estimates.

### Intermediate: Customer Churn Prediction
A classification project for predicting customer churn using telecom data. It includes preprocessing, model training, and an interactive app.

- `data/telco_churn.csv`: customer churn dataset.
- `src/preprocessing.py`: data cleaning and preprocessing pipeline.
- `src/train.py`: trains a churn model with class balancing.
- `src/evaluate.py`: evaluates model performance on holdout data.
- `src/feature_importance.py`: inspects model feature contributions.
- `app/app.py`: Streamlit app for entering customer account details and predicting churn probability.

## How to use this repository

1. Open a terminal in the repository root.
2. Create or activate a Python virtual environment.
3. Install dependencies from the selected project folder, for example:

```bash
cd "Intermediate/Customer Churn Prediction"
pip install -r requirements.txt
```

4. Follow the project-specific README file for data preparation and usage.

## Notes

- Each project is designed to run independently.
- Project-specific dependencies are listed in each `requirements.txt`.
- Training artifacts are stored in the project `model/` folder when created.
- The root tree above reflects the current repository contents.
