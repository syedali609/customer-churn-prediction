import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load("/Users/syedalihussain/Documents/customer_churn/models/saved_models/xgb_model.pkl")

# Page title
st.title("Customer Churn Prediction")
st.write("Fill in the customer details below to predict if they will churn.")

# --- Input Fields ---

# Numerical inputs
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)

# Categorical inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# --- Encode inputs same way as training ---
def encode(val, mapping):
    return mapping.get(val, 0)

gender_enc = encode(gender, {"Male": 1, "Female": 0})
senior_enc = encode(senior_citizen, {"Yes": 1, "No": 0})
partner_enc = encode(partner, {"Yes": 1, "No": 0})
dependents_enc = encode(dependents, {"Yes": 1, "No": 0})
phone_enc = encode(phone_service, {"Yes": 1, "No": 0})
multiple_enc = encode(multiple_lines, {"Yes": 2, "No": 0, "No phone service": 1})
internet_enc = encode(internet_service, {"DSL": 0, "Fiber optic": 1, "No": 2})
security_enc = encode(online_security, {"No": 0, "No internet service": 1, "Yes": 2})
backup_enc = encode(online_backup, {"No": 0, "No internet service": 1, "Yes": 2})
device_enc = encode(device_protection, {"No": 0, "No internet service": 1, "Yes": 2})
tech_enc = encode(tech_support, {"No": 0, "No internet service": 1, "Yes": 2})
tv_enc = encode(streaming_tv, {"No": 0, "No internet service": 1, "Yes": 2})
movies_enc = encode(streaming_movies, {"No": 0, "No internet service": 1, "Yes": 2})
contract_enc = encode(contract, {"Month-to-month": 0, "One year": 1, "Two year": 2})
paperless_enc = encode(paperless_billing, {"Yes": 1, "No": 0})
payment_enc = encode(payment_method, {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3
})

# Combine all inputs into one array
input_data = np.array([[
    gender_enc, senior_enc, partner_enc, dependents_enc,
    phone_enc, multiple_enc, internet_enc, security_enc,
    backup_enc, device_enc, tech_enc, tv_enc, movies_enc,
    contract_enc, paperless_enc, payment_enc,
    tenure, monthly_charges, total_charges
]])

# --- Predict Button ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("---")

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN")
        st.write(f"Churn Probability: **{probability:.0%}**")
    else:
        st.success(f"✅ This customer is likely to STAY")
        st.write(f"Churn Probability: **{probability:.0%}**")