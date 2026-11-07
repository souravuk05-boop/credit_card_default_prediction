
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Risk Prediction")
st.write(
    "An educational machine learning application "
    "for predicting next-month credit card default."
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "best_credit_risk_model.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

st.subheader("Customer Information")

limit_bal = st.number_input(
    "Credit limit (NTD)",
    min_value=1,
    value=100000,
    step=10000
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

sex = st.selectbox(
    "Sex (dataset category)",
    options=[1, 2],
    format_func=lambda x: {
        1: "1 - Male",
        2: "2 - Female"
    }[x]
)

education = st.selectbox(
    "Education",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Graduate school",
        2: "University",
        3: "High school",
        4: "Other / unknown"
    }[x]
)

marriage = st.selectbox(
    "Marital status",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Unknown",
        1: "Married",
        2: "Single",
        3: "Other"
    }[x]
)

st.subheader("Repayment History")

st.caption(
    "Enter the six historical repayment-status "
    "codes used in the UCI dataset."
)

pay_cols = [
    "PAY_1", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

pay_values = {}

for col in pay_cols:
    pay_values[col] = st.selectbox(
        f"{col} repayment status",
        options=list(range(-2, 10)),
        index=2,
        key=col
    )

st.subheader("Billing and Payment History")

bill_values = {}
payment_values = {}

col1, col2 = st.columns(2)

with col1:
    for i in range(1, 7):
        bill_values[f"BILL_AMT{i}"] = st.number_input(
            f"Bill amount {i} (NTD)",
            value=10000.0,
            step=1000.0
        )

with col2:
    for i in range(1, 7):
        payment_values[f"PAY_AMT{i}"] = st.number_input(
            f"Payment amount {i} (NTD)",
            min_value=0.0,
            value=2000.0,
            step=1000.0
        )

if st.button("Predict Default Risk", type="primary"):

    customer_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay_values,
        **bill_values,
        **payment_values
    }

    customer = pd.DataFrame([customer_data])

    # Reproduce training feature engineering
    customer["months_with_delay"] = (
        customer[pay_cols] > 0
    ).sum(axis=1)

    customer["max_delay"] = (
        customer[pay_cols].clip(lower=0).max(axis=1)
    )

    try:
        # Align inputs with the fitted preprocessing pipeline
        expected_columns = (
            model.named_steps["preprocessor"]
            .feature_names_in_
        )

        customer = customer[list(expected_columns)]

        probability = model.predict_proba(
            customer
        )[0, 1]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated next-month default probability",
            f"{probability:.1%}"
        )

        st.progress(
            min(max(float(probability), 0.0), 1.0)
        )

        st.info(
            "This score is a model estimate, not "
            "a lending decision or a validated "
            "measure of individual creditworthiness."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "Educational demonstration using historical "
    "Taiwanese credit card data. Not suitable "
    "for real-world lending decisions."
)
