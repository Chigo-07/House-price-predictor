# app.py
import streamlit as st
import joblib
import numpy as np

# 1. Load the saved model
# Use @st.cache_resource to load the model only once and keep it in memory
@st.cache_resource
def load_model():
    return joblib.load('model/house_price_model.pkl')

model = load_model()

# Page Title and Description
st.title("🏠 House Price Prediction System")
st.write("""
This system predicts house prices based on features from the 'House Prices' dataset.
Adjust the values below to get a prediction.
""")

st.divider()

# 2. User Input Form
st.subheader("Enter House Details")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
    year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=2000)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", options=[0, 1, 2, 3, 4], index=2)
    full_bath = st.selectbox("Full Bathrooms", options=[0, 1, 2, 3, 4], index=1)
    total_bsmt_sf = st.number_input("Total Basement (sq ft)", min_value=0, max_value=6000, value=1000)

# 3. Predict Button
if st.button("Predict Price", type="primary"):
    # Prepare input data as a 2D array matching the training format
    input_data = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]])
    
    # 4. Make Prediction
    prediction = model.predict(input_data)
    
    # Display Result
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")