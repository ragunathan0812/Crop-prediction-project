import streamlit as st
import numpy as np
import joblib

# Load model and label encoder
model = joblib.load("models/crop_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

st.title("🌾 Crop Decision Support System (India)")

st.write("Enter soil nutrients and climate details to get crop recommendation.")

# User Inputs
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=50.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=50.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=400.0, value=100.0)

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = label_encoder.inverse_transform(prediction)
    
    st.success(f"Recommended Crop: 🌱 {crop_name[0].capitalize()}")
