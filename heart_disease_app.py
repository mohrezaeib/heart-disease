"""
Streamlit app: Heart-Disease Risk Predictor
Launch with →  streamlit run heart_disease_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------
MODEL_PATH = Path("output/logreg_pipeline.pkl")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run train_logistic.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Heart-Disease Risk Predictor 💓")
st.write(
    "Enter the patient's data and click **Predict** to estimate the probability "
    "of having heart disease."
)

col1, col2 = st.columns(2)
with col1:
    age     = st.number_input("Age",               min_value=1,   max_value=120, value=50)
    bp      = st.number_input("Blood Pressure",    min_value=40,  max_value=250, value=120)
    hr      = st.number_input("Heart Rate",        min_value=30,  max_value=220, value=80)
with col2:
    gender  = st.selectbox("Gender (0 = female, 1 = male)", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    chol    = st.number_input("Cholesterol",       min_value=50,  max_value=600, value=200)
    qpf     = st.number_input("Quantum Pattern Feature", min_value=0.0, max_value=20.0, value=8.0, step=0.01)

if st.button("Predict"):
    # Assemble single-row dataframe in correct column order
    input_dict = {
        "Age": [age],
        "Gender": [gender],
        "BloodPressure": [bp],
        "Cholesterol": [chol],
        "HeartRate": [hr],
        "QuantumPatternFeature": [qpf],
    }
    X_new = pd.DataFrame(input_dict)

    # Predict
    prob = model.predict_proba(X_new)[0, 1]
    pred = model.predict(X_new)[0]

    st.subheader("Results")
    st.write(f"Predicted **probability of Heart Disease**: **{prob:.2%}**")
    st.write(f"Predicted class: {'Disease (1)' if pred==1 else 'No Disease (0)'}")

    st.progress(prob)
