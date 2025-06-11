import streamlit as st
import joblib
import pandas as pd

# Load model and feature columns
linreg = joblib.load('models/car_insurance_linreg.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

st.title("Car Insurance Premium Predictor")

st.write("Enter the driver's and car's information:")

# Collect user input
inputs = {}
for feature in feature_cols:
    if feature == 'Previous Accidents' or feature == 'Car Age' or feature == 'Driver Experience':
        val = st.number_input(feature, min_value=0, value=0)
    elif feature == 'Annual Mileage (x1000 km)':
        val = st.number_input(feature, min_value=1, value=15)
    else:  # Driver Age
        val = st.number_input(feature, min_value=16, value=30)
    inputs[feature] = val

# Prediction
if st.button("Predict Insurance Premium"):
    input_df = pd.DataFrame([inputs])[feature_cols]
    pred = linreg.predict(input_df)[0]
    st.success(f"Estimated Insurance Premium: **${pred:.2f}**")
