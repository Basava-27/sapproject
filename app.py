import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained objects
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# ---------------------- UI ----------------------
st.title("🚨 Bank Transaction Fraud Detector")

st.markdown("Enter transaction details below to check if it's fraudulent.")

# 🧾 User Inputs
user_input = {}
columns_needed = [
    'Customer_ID', 'Customer_Name', 'Gender', 'Age', 'State', 'City', 'Bank_Branch', 
    'Account_Type', 'Transaction_ID', 'Transaction_Date', 'Transaction_Time', 
    'Transaction_Amount', 'Merchant_ID', 'Transaction_Type', 'Merchant_Category', 
    'Account_Balance', 'Transaction_Device', 'Transaction_Location', 'Device_Type', 
    'Transaction_Currency', 'Customer_Contact', 'Transaction_Description', 'Customer_Email'
]

# Dynamically create input fields
for col in columns_needed:
    if col in ['Age', 'Transaction_Amount', 'Account_Balance']:
        user_input[col] = st.number_input(col, step=1.0)
    else:
        user_input[col] = st.text_input(col)

if st.button("Detect Fraud"):
    input_df = pd.DataFrame([user_input])

    # Encode categorical values
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except:
                st.warning(f"Unknown value for '{col}': {input_df[col].values[0]}")
                st.stop()

    # Scale numeric values
    numeric_cols = scaler.feature_names_in_
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"❌ Fraudulent Transaction Detected! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Transaction is Legitimate (Confidence: {1 - proba:.2f})")
