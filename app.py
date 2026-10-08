import streamlit as st
import pandas as pd
import joblib

# ------------------------------------
# Load Model
# ------------------------------------
model = joblib.load("fraud_model.pkl")

# ------------------------------------
# Page Config
# ------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ------------------------------------
# Styling
# ------------------------------------
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    font-size: 18px;
    color: gray;
}
.card {
    padding: 20px;
    border-radius: 14px;
    background-color: #f4f6f9;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# Header
# ------------------------------------
st.markdown('<div class="title">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-world ML system using synthetic transaction data</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------
# Input Form
# ------------------------------------
st.subheader("🧪 Enter Transaction Details")

with st.form("txn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("💰 Amount", min_value=0.0, value=1200.0)
        hour = st.slider("🕒 Transaction Hour", 0, 23, 14)
        txn_frequency = st.slider("🔁 Transactions in last hour", 1, 10, 2)

    with col2:
        device = st.selectbox("📱 Device Used", ["Mobile", "Desktop"])
        merchant_risk = st.selectbox("🏪 Merchant Risk", ["Low", "Medium", "High"])
        prev_failed_txn = st.slider("❌ Previous Failed Transactions", 0, 10, 0)

    with col3:
        is_online = st.selectbox("🌐 Transaction Type", ["Online", "Offline"])
        is_international = st.selectbox("🌍 Location", ["Domestic", "International"])

    submitted = st.form_submit_button("🔍 Check Fraud Risk")

# ------------------------------------
# Prediction
# ------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "amount": amount,
        "hour": hour,
        "is_international": 1 if is_international == "International" else 0,
        "is_online": 1 if is_online == "Online" else 0,
        "device": device,
        "merchant_risk": merchant_risk,
        "prev_failed_txn": prev_failed_txn,
        "txn_frequency": txn_frequency
    }])

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.markdown("---")
    st.subheader("🧠 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("🚨 Fraudulent Transaction")
        else:
            st.success("✅ Legitimate Transaction")

    with col2:
        st.metric("Fraud Probability", f"{prob*100:.2f}%")

    st.progress(float(prob))

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.caption("🔐 Fraud Detection System | Synthetic Data | ML + Streamlit")
