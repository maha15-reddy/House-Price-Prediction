
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page title
st.title("🏠 House Price Prediction")

st.write("Enter the house details below to predict the house price.")

# Input fields
longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")

# Prediction
if st.button("Predict Price"):

    data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income]
    })

    # Scale input
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")
