import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------- Initial Setup ----------------------

# Ensure xgboost is available if not pre-installed
try:
    import xgboost
except ImportError:
    os.system("pip install xgboost")
    import xgboost

# Load model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    return model, scaler, label_encoders

model, scaler, label_encoders = load_model_and_preprocessors()

# ---------------------- UI Layout ----------------------

st.set_page_config(page_title="Bank Fraud Detector", page_icon="💳")
st.title("🚨 Bank Transaction Fraud Detector")
st.markdown("Use the form below to input transaction details and detect if it's **fraudulent**.")

# ---------------------- User Input ----------------------

user_input = {}
columns_needed = [
    'Customer_ID', 'Customer_Name', 'Gender', 'Age', 'State', 'City', 'Bank_Branch',
    'Account_Type', 'Transaction_ID', 'Transaction_Date', 'Transaction_Time',
    'Transaction_Amount', 'Merchant_ID', 'Transaction_Type', 'Merchant_Category',
    'Account_Balance', 'Transaction_Device', 'Transaction_Location', 'Device_Type',
    'Transaction_Currency', 'Customer_Contact', 'Transaction_Description', 'Customer_Email'
]

# Friendly UI rendering
st.subheader("📋 Transaction Details")
for col in columns_needed:
    if col in ['Age', 'Transaction_Amount', 'Account_Balance']:
        user_input[col] = st.number_input(f"{col}", step=1.0)
    else:
        user_input[col] = st.text_input(f"{col}")

# ---------------------- Prediction ----------------------

if st.button("🔍 Detect Fraud"):
    input_df = pd.DataFrame([user_input])

    try:
        # Encode categorical columns
        for col, le in label_encoders.items():
            if col in input_df.columns:
                if input_df[col].values[0] not in le.classes_:
                    st.warning(f"⚠️ '{input_df[col].values[0]}' is an unseen value for '{col}'.")
                    st.stop()
                input_df[col] = le.transform(input_df[col])

        # Scale numeric columns
        numeric_cols = scaler.feature_names_in_
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Make prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        # Output
        if prediction == 1:
            st.error(f"❌ Fraudulent Transaction Detected!\n\nConfidence: **{proba:.2%}**")
        else:
            st.success(f"✅ Transaction is Legitimate.\n\nConfidence: **{(1 - proba):.2%}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")
