# Customer Churn Prediction

Binary classification model built on telecom customer data to predict customer churn, with a focus on handling class imbalance, model interpretability, and robust evaluation. Includes a Streamlit app for interactive predictions.

## Overview

- Built a binary classification model on telecom customer data to predict churn, applying **SMOTE** to address class imbalance in the target variable.
- Conducted **feature importance analysis** (SHAP) to identify key churn drivers, informing model interpretability and business insights.
- Evaluated model performance using **ROC-AUC** and other classification metrics to ensure robust generalization.

## Dataset

Telco Customer Churn dataset (IBM sample dataset, commonly distributed via Kaggle). ~7,000 customer records with demographic, account, and service-usage features, and a binary `Churn` target (Yes/No). Place the CSV at `data/telco_churn.csv`.

## Project Structure

```
Customer-Churn-Prediction/
├── src/
│   ├── preprocessing.py       # load, clean, encode, split data
│   ├── train.py                # SMOTE + model training, saves model/churn_model.pkl
│   ├── evaluate.py             # ROC-AUC, F1, confusion matrix, ROC curve
│   └── feature_importance.py   # SHAP analysis, saves shap_summary.png
├── app/
│   └── app.py                  # Streamlit app for interactive churn prediction
├── model/                      # saved model artifact + output plots (gitignored)
├── data/                       # dataset CSV (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/priya369-ps/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

## Usage

**1. Train a model**
```bash
python src/train.py --data data/telco_churn.csv --model-out model/churn_model.pkl --model-name xgboost
```

**2. Evaluate it**
```bash
python src/evaluate.py --data data/telco_churn.csv --model model/churn_model.pkl
```

**3. Run feature importance analysis**
```bash
python src/feature_importance.py --data data/telco_churn.csv --model model/churn_model.pkl
```

**4. Launch the demo app**
```bash
streamlit run app/app.py
```

## Approach

1. **EDA** — churn distribution, churn rate by contract type, tenure, monthly charges.
2. **Preprocessing** (`src/preprocessing.py`) — cleaned `TotalCharges`, encoded categoricals, scaled numerics.
3. **Train/test split** — stratified 80/20 split performed *before* any resampling, to avoid data leakage.
4. **Class imbalance handling** (`src/train.py`) — SMOTE applied to the training set only.
5. **Modeling** — Logistic Regression, Random Forest, XGBoost, selectable via `--model-name`.
6. **Evaluation** (`src/evaluate.py`) — ROC-AUC, precision, recall, F1, confusion matrix, ROC curve.
7. **Feature importance** (`src/feature_importance.py`) — SHAP summary plot on the trained tree model.

## Results

| Model               | ROC-AUC | F1-score |
|---------------------|---------|----------|
| Logistic Regression | TBD     | TBD      |
| Random Forest        | TBD     | TBD      |
| XGBoost              | TBD     | TBD      |

*(Fill in after running `src/evaluate.py`.)*

**Top churn drivers identified:** TBD (e.g. contract type, tenure, monthly charges, tech support) — see `model/shap_summary.png` after running feature importance analysis.

## Key Takeaways

- SMOTE was applied strictly to the training set to avoid inflating test performance through data leakage.
- ROC-AUC and F1 were prioritized over raw accuracy, since accuracy is misleading on an imbalanced churn target (~27% positive class).
- SHAP was used over plain feature importances to capture both magnitude and direction of each feature's effect on churn probability.
