# House Price Prediction

This project builds a simple machine learning app to predict house prices using a synthetic housing dataset. It includes:

- a training pipeline with multiple regression models
- model evaluation and selection
- a Streamlit web app for making predictions

## Project Structure

- `app/` - Streamlit web app
- `src/` - training, preprocessing, prediction, and data loading code
- `models/` - saved trained model files
- `data/` - dataset folder (empty unless you add your own data)

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training script:

```bash
python src/train.py
```

This will train several models, compare their performance, and save the best one to `models/best_model.pkl`.

## Run the App

Start the Streamlit app:

```bash
streamlit run app/app.py
```

## Notes

The project currently uses a synthetic housing dataset generated offline, so it does not require downloading external data. This makes it easy to run and reproduce.
