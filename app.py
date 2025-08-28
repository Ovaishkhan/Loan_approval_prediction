import streamlit as st
import joblib
import numpy as np
import pandas as pd
model=joblib.load("loan_model.pkl")


st.title("🏦 Loan Approval Prediction App")

# Collect inputs from user
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Non Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# When user clicks Predict
if st.button("Predict Loan Approval"):

    # Convert categorical values into same encoding as training
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    # Collect features into array
    features = np.array([
        gender, married, education, self_employed,
        applicant_income, loan_amount, loan_amount_term,
        credit_history, property_area
    ]).reshape(1, -1)   # ✅ reshape into 2D array

    # Prediction
    prediction = model.predict(features)[0]

    # Output
    if prediction == 1:
        st.success("🎉 Congratulations! Your loan is likely to be approved.")
    else:
        st.error("❌ Sorry, your loan application is likely to be rejected.")
