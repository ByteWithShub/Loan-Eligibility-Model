#Loan Eligibility Prediction App
import streamlit as st
import joblib
import pandas as pd
from src.preprocessing import clean_data, encode_data

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="🏦", layout="centered")

# Load saved artifacts
model = joblib.load("models/loan_model.pkl")
columns = joblib.load("models/columns.pkl")

st.title("Loan Eligibility Prediction App")
st.markdown("Enter applicant details below to predict whether the loan is likely to be approved.")

st.subheader("Applicant Information")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

st.subheader("Financial Information")

applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
loan_term = st.selectbox("Loan Amount Term", [360, 180, 120, 84, 60, 36, 12], index=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad / Missing (0.0)")
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict Loan Status"):
    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Approved": "Y"
    }

    input_df = pd.DataFrame([input_data])

    input_df = clean_data(input_df)
    input_df = encode_data(input_df)
    input_df = input_df.drop("Loan_Approved", axis=1)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
    else:
        probability = None

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(" Loan Approved")
    else:
        st.error("Sorry :( Loan Not Approved")

    if probability is not None:
        st.info(f"Approval Probability: {probability:.2%}")
