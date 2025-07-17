import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import matplotlib
import streamlit.components.v1 as components
from sklearn.svm import SVC

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
    "Upload patient data or manually enter details to predict heart‑disease risk **and** understand which features drive each prediction through interactive SHAP visualisations."
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
# Helper – SHAP utilities
# ───────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values using a best‑effort strategy that covers pipelines & raw models."""
    # Sample a small background for performance
    background = X.copy()
    if len(background) > 100:
        background = background.sample(n=100, random_state=42)

    # 1️⃣ Generic explainer (works on many model types)
    try:
        explainer = shap.Explainer(model, background)
        return explainer, explainer(X)
    except Exception:
        pass

    print("# 2️⃣ Pipeline fallback – explain final estimator on transformed data")
    if hasattr(model, "named_steps"):
        try:
            bg_trans = model[:-1].transform(background)
            X_trans = model[:-1].transform(X)
            explainer = shap.Explainer(model[-1], bg_trans)
            shap_vals = explainer(X_trans)
            return explainer, shap_vals
        except Exception:
            pass

    st.warning("Unable to compute SHAP values for this model.")
    return None, None


def st_shap_plot(plot_obj, *, height: int = 400):
    """Render either an interactive HTML SHAP plot or a Matplotlib figure."""
    # Interactive plot (has .html method)
    if hasattr(plot_obj, "html") and callable(plot_obj.html):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot_obj.html()}</body>"
        components.html(shap_html, height=height, scrolling=True)
        return

    # Matplotlib figure / axes fallback
    if isinstance(plot_obj, matplotlib.figure.Figure):
        st.pyplot(plot_obj)
    elif hasattr(plot_obj, "figure") and isinstance(plot_obj.figure, matplotlib.figure.Figure):
        st.pyplot(plot_obj.figure)
    else:
        st.write("⚠️ Could not display SHAP plot – unsupported object type.")


def render_shap_plots(shap_values, X: pd.DataFrame):
    """Render an appropriate SHAP visual, handling multi‑output (e.g. binary) explanations."""
    st.subheader("Feature Contribution (SHAP)")
    shap.initjs()

    # For binary classifiers shap_values shape = (n_samples, 2, n_features).
    # Pick the *positive* class (index 1) by default.
    is_multi_output = len(shap_values.shape) == 3
    if is_multi_output:
        shap_values = shap_values[:, 1, :]

    if len(X) == 1:
        st.write("Waterfall plot for this individual prediction:")
        plot_obj = shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st_shap_plot(plot_obj)
    else:
        st.write("Summary feature‑importance across all uploaded rows:")
        plot_obj = shap.plots.bar(shap_values, max_display=10, show=False)
        st_shap_plot(plot_obj, height=500)

# ───────────────────────────────────────────────
# Prediction & SHAP
# ───────────────────────────────────────────────
# Prediction
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
    # Predict class probabilities & labels
    prob = model.predict_proba(input_df)[:, 1]
    preds = model.predict(input_df)

    # Display predictions
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
