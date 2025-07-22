import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("models/fraud_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

st.title("🔍 Fraud Detection for Reservations")
st.markdown("Enter the reservation details below:")

# Define the fields expected (adjust if needed)
input_data = {}

input_data['nbr_place'] = st.number_input("Number of places", min_value=1, max_value=100, value=2)
input_data['number_of_voyageurs'] = st.number_input("Number of travellers", min_value=1, max_value=50, value=2)
input_data['Status'] = st.selectbox("Reservation Status", ["confirmé", "annulé", "en attente"])
input_data['reminderSent'] = st.selectbox("Reminder Sent?", [True, False])
input_data['volId_frequency'] = st.number_input("Flight frequency", value=30)
input_data['account_age_days'] = st.number_input("Account age (days)", value=10)
input_data['payment_delay_days'] = st.number_input("Payment delay (days)", value=0)
input_data['total_payment_amount'] = st.number_input("Total payment amount", value=100.0)
input_data['payment_failures_count'] = st.number_input("Payment failure count", value=0)
input_data['payment_status'] = st.selectbox("Payment Status", ["réussi", "échoué"])
input_data['Pays'] = st.selectbox("Country", ["France", "Tunisie", "Algérie", "Autre"])
input_data['Ville'] = st.text_input("City", "Paris")
input_data['newsletter_abonne'] = st.selectbox("Subscribed to newsletter?", [True, False])
input_data['satisfaction_client'] = st.slider("Client satisfaction (1-5)", 1, 5, 3)
input_data['annulations_precedentes'] = st.number_input("Previous cancellations", value=0)
input_data['modifications_reservation'] = st.number_input("Modifications", value=0)
input_data['tentatives_paiement'] = st.number_input("Payment attempts", value=1)
input_data['email_domain'] = st.text_input("Email domain", "gmail.com")
input_data['suspicious_email_domain'] = st.selectbox("Suspicious email domain?", [True, False])

if st.button("🔍 Predict Fraud"):
    df_input = pd.DataFrame([input_data])

    # Encode categorical variables
    for col, le in label_encoders.items():
        if df_input[col].iloc[0] not in le.classes_:
            le.classes_ = np.append(le.classes_, df_input[col].iloc[0])
        df_input[col] = le.transform(df_input[col])

    # Predict
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    if prediction == 1:
        st.error(f"🚨 This reservation is likely **fraudulent**. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ This reservation seems legitimate. (Probability of fraud: {probability:.2%})")
