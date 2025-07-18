import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from shap import Explanation
import matplotlib
import streamlit.components.v1 as components

# Nicer looking charts in Streamlit
plt.rcParams["figure.dpi"] = 120
st.set_page_config(page_title="Heart‑Disease Risk Predictor", layout="centered")

# ───────────────────────────────────────────────
# Load available models
# ───────────────────────────────────────────────
MODEL_DIR = Path("output")
ALL_MODELS = {p.stem.removeprefix("model_"): p for p in MODEL_DIR.glob("model_*.pkl")}
MODEL_BASE_NAMES = sorted({name.replace("_no_qpf", "") for name in ALL_MODELS})

@st.cache_resource
def load_model(base_name: str, *, has_qpf: bool = True):
    """Load a pre‑trained sklearn pipeline by name."""
    suffix = "" if has_qpf else "_no_qpf"
    key = f"{base_name}{suffix}"
    if key not in ALL_MODELS:
        st.error(f"Model file not found: {key}")
        st.stop()
    return joblib.load(ALL_MODELS[key])

# ───────────────────────────────────────────────
# UI – header & sidebar
# ───────────────────────────────────────────────
st.title("Heart‑Disease Risk Predictor 💓")
st.write(
    "Upload patient data or manually enter details to predict heart‑disease risk "
    "**and** understand which features drive each prediction through SHAP visualisations."
)

selected_model = st.sidebar.selectbox("Select a Trained Model", MODEL_BASE_NAMES)
use_file = st.checkbox("Upload CSV with patient data")

# ───────────────────────────────────────────────
# Input
# ───────────────────────────────────────────────
if use_file:
    uploaded = st.file_uploader(
        "Upload CSV with columns: Age, Gender, BloodPressure, Cholesterol, HeartRate, QuantumPatternFeature",
        type=["csv"],
    )
    if uploaded is None:
        st.stop()

    input_df = pd.read_csv(uploaded)

    # Remove target column if accidentally included
    if "HeartDisease" in input_df.columns:
        input_df = input_df.drop(columns=["HeartDisease"])

    expected_cols = ["Age", "Gender", "BloodPressure", "Cholesterol", "HeartRate"]
    has_qpf = "QuantumPatternFeature" in input_df.columns
    if has_qpf:
        expected_cols.append("QuantumPatternFeature")

    missing = set(expected_cols) - set(input_df.columns)
    if missing:
        st.error(f"Uploaded file is missing required columns: {missing}")
        st.stop()
else:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        bp = st.number_input("Blood Pressure", 40, 250, 120)
        hr = st.number_input("Heart Rate", 30, 220, 80)
    with col2:
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        chol = st.number_input("Cholesterol", 50, 600, 200)
        qpf = st.number_input("Quantum Pattern Feature (optional)", 0.0, 20.0, np.nan)

    input_dict = {
        "Age": [age],
        "Gender": [gender],
        "BloodPressure": [bp],
        "Cholesterol": [chol],
        "HeartRate": [hr],
    }
    has_qpf = not np.isnan(qpf)
    if has_qpf:
        input_dict["QuantumPatternFeature"] = [qpf]

    input_df = pd.DataFrame(input_dict)

# ───────────────────────────────────────────────
# SHAP Utilities
# ───────────────────────────────────────────────
def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values using LinearExplainer with support for different pipeline styles."""
    try:
        steps = model.named_steps

        # Try standard scaler-based pipelines
        if "scaler" in steps and "clf" in steps:
            scaler = steps["scaler"]
            clf = steps["clf"]
            X_scaled = scaler.transform(X)
            background = np.zeros_like(X_scaled)
            explainer = shap.LinearExplainer(clf, background, feature_names=X.columns)
            shap_values = explainer(X_scaled)
            return explainer, shap_values

        # Try pipelines with a full preprocessor (e.g., prep) before clf
        elif "prep" in steps and "clf" in steps:
            prep = steps["prep"]
            clf = steps["clf"]
            X_trans = prep.transform(X)
            background = np.zeros_like(X_trans)
            explainer = shap.LinearExplainer(clf, background, feature_names=getattr(prep, "get_feature_names_out", lambda: X.columns)())
            shap_values = explainer(X_trans)
            return explainer, shap_values

        else:
            st.warning("Unsupported pipeline structure for SHAP.")
            return None, None

    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
        return None, None

def render_shap_plots(shap_values, X: pd.DataFrame):
    """Render SHAP force and bar plots using Explanation object."""
    st.subheader("Feature Contribution (SHAP)")

    st.write("🔎 **SHAP Force Plot (All Features)**")
    try:
        fig_force = shap.plots.force(shap_values[0], matplotlib=True, show=False)
        st.pyplot(fig_force)
    except Exception as e:
        st.warning(f"Could not render SHAP force plot: {e}")

    st.write("📊 **SHAP Bar Chart**")
    try:
        expl = Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=X.values[0],
            feature_names=X.columns.tolist()
        )
        fig_bar, ax = plt.subplots()
        shap.plots.bar(expl, show=False)
        st.pyplot(fig_bar)
    except Exception as e:
        st.warning(f"Could not render SHAP bar plot: {e}")

# ───────────────────────────────────────────────
# Prediction & SHAP
# ───────────────────────────────────────────────
predict_clicked = st.button("Predict")
explain_clicked = st.button(
    "Explain with SHAP",
    disabled=not selected_model.startswith("logreg_basic")
)

if predict_clicked or explain_clicked:
    model = load_model(selected_model, has_qpf=has_qpf)

    # Align input features with model expectations
    try:
        prep = model.named_steps.get("prep") if hasattr(model, "named_steps") else None
        expected_cols = (
            input_df.columns if prep == "passthrough" else getattr(prep, "feature_names_in_", input_df.columns)
        )
        input_df = input_df.reindex(columns=expected_cols, fill_value=0)
    except Exception as e:
        st.warning(f"Could not align features automatically: {e}")

if predict_clicked:
    prob = model.predict_proba(input_df)[:, 1]
    preds = model.predict(input_df)

    st.subheader("Prediction Results")
    if len(input_df) == 1:
        st.write(f"**Predicted Probability:** `{prob[0]:.2%}`")
        st.write(f"**Predicted Class:** `{'Disease (1)' if preds[0] else 'No Disease (0)'}`")
        st.progress(float(prob[0]))
    else:
        results_df = input_df.copy()
        results_df["Pred_Prob"] = prob
        results_df["Pred_Class"] = preds
        st.dataframe(results_df.reset_index(drop=True))

if explain_clicked:
    explainer, shap_values = compute_shap_values(model, input_df)
    if shap_values is not None:
        render_shap_plots(shap_values, input_df)
