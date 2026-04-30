import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📞",
    layout="centered"
)

# ── Load model & preprocessor ────────────────────────────────
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    model        = joblib.load("model.pkl")
    return preprocessor, model

preprocessor, model = load_artifacts()

# ── Header ───────────────────────────────────────────────────
st.title("📞 Customer Churn Predictor")
st.markdown(
    "**LightGBM model** trained on Telco data · ROC-AUC **0.82**  \n"
    "Fill in the customer details below and click **Predict**."
)
st.divider()

# ── Input form ───────────────────────────────────────────────
st.subheader("👤 Customer Demographics")
col1, col2, col3 = st.columns(3)
with col1:
    gender        = st.selectbox("Gender",       ["Male", "Female"])
with col2:
    senior        = st.selectbox("Senior Citizen", ["No", "Yes"])
    SeniorCitizen = 1 if senior == "Yes" else 0
with col3:
    partner       = st.selectbox("Partner",      ["No", "Yes"])

col4, col5 = st.columns(2)
with col4:
    dependents    = st.selectbox("Dependents",   ["No", "Yes"])
with col5:
    tenure        = st.slider("Tenure (months)", 0, 72, 12)

st.divider()
st.subheader("📱 Services")

col6, col7 = st.columns(2)
with col6:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
with col7:
    multiple_lines = st.selectbox("Multiple Lines",
                                  ["No", "Yes", "No phone service"])

col8, col9 = st.columns(2)
with col8:
    internet_service = st.selectbox("Internet Service",
                                    ["DSL", "Fiber optic", "No"])
with col9:
    online_security  = st.selectbox("Online Security",
                                    ["No", "Yes", "No internet service"])

col10, col11 = st.columns(2)
with col10:
    online_backup    = st.selectbox("Online Backup",
                                    ["No", "Yes", "No internet service"])
with col11:
    device_protection = st.selectbox("Device Protection",
                                     ["No", "Yes", "No internet service"])

col12, col13 = st.columns(2)
with col12:
    tech_support  = st.selectbox("Tech Support",
                                 ["No", "Yes", "No internet service"])
with col13:
    streaming_tv  = st.selectbox("Streaming TV",
                                 ["No", "Yes", "No internet service"])

streaming_movies = st.selectbox("Streaming Movies",
                                ["No", "Yes", "No internet service"])

st.divider()
st.subheader("💳 Account Info")

col14, col15 = st.columns(2)
with col14:
    contract    = st.selectbox("Contract",
                               ["Month-to-month", "One year", "Two year"])
with col15:
    paperless   = st.selectbox("Paperless Billing", ["No", "Yes"])

payment_method = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

col16, col17 = st.columns(2)
with col16:
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
with col17:
    total_charges   = st.number_input("Total Charges ($)",   0.0, 9000.0,
                                      float(monthly_charges * tenure) if tenure > 0 else 0.0,
                                      step=1.0)

st.divider()

# ── Predict ──────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    input_df = pd.DataFrame([{
        "gender":           gender,
        "SeniorCitizen":    SeniorCitizen,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }])

    X_prep      = preprocessor.transform(input_df)
    churn_prob  = model.predict_proba(X_prep)[0][1]
    churn_label = "⚠️ Will Churn" if churn_prob >= 0.5 else "✅ Will Not Churn"

    # Result card
    if churn_prob >= 0.5:
        st.error(f"### {churn_label}")
        st.error(f"**Churn Probability: {churn_prob:.1%}**  \nThis customer is at high risk. Consider retention offers.")
    else:
        st.success(f"### {churn_label}")
        st.success(f"**Churn Probability: {churn_prob:.1%}**  \nThis customer is likely to stay.")

    # Probability bar
    st.markdown("#### Risk Meter")
    st.progress(float(churn_prob))
    st.caption(f"Churn probability: {churn_prob:.1%}")

    # Key factors (simple interpretation)
    st.markdown("---")
    st.markdown("#### 📋 Summary of Inputs")
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Shubh Gupta · VIT Bhopal CSE ")
