# app.py
import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 House Price Prediction System")
st.write("Enter the details of the house below to estimate its price.")

# --- 1. Load the Saved Model ---
# Feedback: Add try/except for model loading
try:
    model_path = 'model/house_price_model.pkl'
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please run 'model_development.py' first.")
        st.stop()
    
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- 2. User Input Features ---
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    year_built = st.number_input("Year Built", 1900, 2024, 2000)

with col2:
    total_bsmt_sf = st.number_input("Total Basement (sq ft)", 0, 3000, 1000)
    garage_cars = st.selectbox("Garage Cars", [0, 1, 2, 3, 4])
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4])

# --- 3 & 4. Prediction ---
if st.button("Predict Price"):
    # Prepare input array (must match the order in training)
    input_data = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]])
    
    # Feedback: Add try/except for prediction
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")