import sys
from pathlib import Path

# Allow importing from the src/ folder
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
for path in [str(ROOT_DIR), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.append(path)

import streamlit as st

try:
    from src.predict import load_model, predict_price
except ModuleNotFoundError:
    from predict import load_model, predict_price

st.title("🏠 House Price Predictor")
st.write("Enter the house details below to get a predicted price")

model_path = ROOT_DIR / "models" / "best_model.pkl"
model = None

if model_path.exists():
    try:
        model = load_model(model_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
else:
    st.warning(
        f"The trained model file was not found at {model_path}. "
        "Train the model first and then run the app again."
    )

med_inc = st.slider("Median Income (in ₹10,000s)", 1.0, 15.0, 5.0)
house_age = st.slider("House Age (years)", 1, 52, 20)
ave_rooms = st.slider("Average Rooms per Household", 2.0, 10.0, 6.0)
ave_bedrms = st.slider("Average Bedrooms per Household", 0.5, 3.0, 1.2)
population = st.slider("Population", 3, 5000, 1000)
ave_occup = st.slider("Average Occupants per Household", 1.0, 6.0, 3.0)
latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
longitude = st.slider("Longitude", -124.0, -114.0, -118.0)

if st.button("Predict Price"):
    if model is None:
        st.info("The model is not available yet. Please train it first.")
    else:
        input_data = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude
        }
        price = predict_price(model, input_data)
        st.success(f"The predicted house price is: ₹{price * 100000:.2f}")