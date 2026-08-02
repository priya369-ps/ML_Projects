"""
Streamlit app for the Customer Churn Prediction project.

Run with:
    streamlit run app/app.py
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter a customer's details to predict their probability of churning.")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

if bundle is None:
    st.warning(
        "No trained model found at `model/churn_model.pkl`. "
        "Run `python src/train.py --data data/telco_churn.csv` first."
    )
else:
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"],
            )
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"],
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"],
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"],
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total Charges", 0.0, 10000.0, 840.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # NOTE: fill in remaining columns with dataset-typical defaults as needed
        # so this matches the exact schema the model was trained on.
        input_df = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )

        try:
            X_proc = preprocessor.transform(input_df)
            proba = model.predict_proba(X_proc)[0][1]
            pred = "Likely to Churn" if proba >= 0.5 else "Likely to Stay"

            st.subheader(pred)
            st.metric("Churn Probability", f"{proba:.1%}")
        except Exception as e:
            st.error(
                "Input columns don't fully match the training schema. "
                f"Ensure the form covers all required features. ({e})"
            )
