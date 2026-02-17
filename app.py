import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AI-Based Crop Recommendation System")
st.markdown("Provide soil and climate conditions to get the best crop suggestion.")

# -------------------------------------------------
# LOAD MODEL FILES
# -------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("models/random_forest_crop_recommendation.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    metadata = joblib.load("models/model_metadata.pkl")
    return model, encoder, feature_names, metadata

model, encoder, feature_names, metadata = load_models()

# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.header("🧪 Enter Soil & Climate Data")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 145, 50)
K = st.sidebar.slider("Potassium (K)", 0, 205, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

state = st.sidebar.selectbox(
    "Select State",
    ["Punjab", "Maharashtra", "Tamil Nadu", "Karnataka", "Uttar Pradesh"]
)

# -------------------------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------------------------
input_data = pd.DataFrame(
    [[N, P, K, temperature, humidity, ph, rainfall]],
    columns=feature_names
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("🌱 Recommend Crop"):

    prediction_encoded = model.predict(input_data)[0]
    prediction = encoder.inverse_transform([prediction_encoded])[0]

    probabilities = model.predict_proba(input_data)[0]
    confidence = round(np.max(probabilities) * 100, 2)

    st.success(f"🌾 Recommended Crop: **{prediction}**")
    st.info(f"🔍 Confidence Level: **{confidence}%**")

    # -------------------------------------------------
    # TOP 3 PREDICTIONS
    # -------------------------------------------------
    st.subheader("📊 Top 3 Crop Suggestions")

    top_3_indices = probabilities.argsort()[-3:][::-1]

    for i, idx in enumerate(top_3_indices, 1):
        crop_name = encoder.classes_[idx]
        prob = probabilities[idx]
        st.write(f"{i}. {crop_name} — {prob*100:.2f}%")

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.subheader("📈 Feature Importance")

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
ax.barh(feature_importance["Feature"], feature_importance["Importance"])
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)

# -------------------------------------------------
# STATE CONTEXT
# -------------------------------------------------
st.subheader("🌍 State Crop Context")

if state == "Punjab":
    st.write("Common crops: Rice, Wheat")
elif state == "Maharashtra":
    st.write("Common crops: Cotton, Pulses")
elif state == "Tamil Nadu":
    st.write("Common crops: Rice, Banana")
elif state == "Karnataka":
    st.write("Common crops: Ragi, Maize")
elif state == "Uttar Pradesh":
    st.write("Common crops: Sugarcane, Wheat")

# -------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------
st.subheader("📌 Model Information")

st.write(f"• Model Type: {metadata['model_type']}")
st.write(f"• Accuracy: {metadata['accuracy'] * 100:.2f}%")
st.write(f"• Number of Features: {metadata['n_features']}")
st.write(f"• Number of Crop Classes: {metadata['n_classes']}")
st.write(f"• Best Parameters: {metadata['best_params']}")

st.markdown("---")
st.markdown("🚀 Developed using Machine Learning for Indian Agriculture")
