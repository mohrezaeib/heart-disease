"""
Train basic and feature-engineered models with and without QuantumPatternFeature.
Saves pipelines for later inference and SHAP explanation.
"""

import pandas as pd
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Paths
CSV_PATH = Path("data/Heart Prediction Quantum Dataset.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)
y = df["HeartDisease"]
X_basic = df.drop(columns=["HeartDisease"])

# Feature Engineering
# df_fe = df.copy()
# df_fe["bp_age_ratio"] = df_fe["BloodPressure"] / df_fe["Age"]
# df_fe["age_bp_interaction"] = df_fe["Age"] * df_fe["BloodPressure"]
# df_fe["age_group_bin"] = pd.cut(df_fe["Age"], bins=[0, 45, 60, 120], labels=[0, 1, 2]).astype(int)
# df_fe = pd.get_dummies(df_fe, columns=["age_group_bin"], drop_first=True)

# y_fe = df_fe["HeartDisease"]
# X_fe = df_fe.drop(columns=["HeartDisease"])

# Drop QPF version
X_basic_no_qpf = X_basic.drop(columns=["QuantumPatternFeature"])
# X_fe_no_qpf = X_fe.drop(columns=["QuantumPatternFeature"])

# Model factory
def make_pipeline(model, feature_names, use_scaling=True):
    steps = []
    if use_scaling:
        scaler = ColumnTransformer([
            ("scale", StandardScaler(), feature_names)
        ], remainder="passthrough")
        steps.append(("prep", scaler))
    else:
        steps.append(("prep", "passthrough"))
    steps.append(("clf", model))
    return Pipeline(steps)

# Define model configs
models = {
    "logreg_basic": (LogisticRegression(max_iter=1000, solver="liblinear"), True, X_basic, y),
    "random_forest": (RandomForestClassifier(n_estimators=200, random_state=42), False, X_basic, y),
    "svm": (SVC(kernel="rbf", probability=True), True, X_basic, y),
    "decision_tree": (DecisionTreeClassifier(random_state=42), False, X_basic, y),

    # "logreg_eng": (LogisticRegression(max_iter=1000, solver="liblinear"), True, X_fe, y_fe),
    # "random_forest_eng": (RandomForestClassifier(n_estimators=200, random_state=42), False, X_fe, y_fe),
    # "svm_eng": (SVC(kernel="rbf", probability=True), True, X_fe, y_fe),
    # "decision_tree_eng": (DecisionTreeClassifier(random_state=42), False, X_fe, y_fe),
}

# Train and save each model + no_qpf version
for name, (clf, scale, X_data, y_data) in models.items():
    feat_names = X_data.columns.tolist()

    # With QPF
    model = make_pipeline(clf, feat_names, use_scaling=scale)
    model.fit(X_data, y_data)
    joblib.dump(model, OUT_DIR / f"model_{name}.pkl")

    # Without QPF
    if "QuantumPatternFeature" in X_data.columns:
        X_no_qpf = X_data.drop(columns=["QuantumPatternFeature"])
        model_nq = make_pipeline(clf, X_no_qpf.columns.tolist(), use_scaling=scale)
        model_nq.fit(X_no_qpf, y_data)
        joblib.dump(model_nq, OUT_DIR / f"model_{name}_no_qpf.pkl")

print("✅ All models trained and saved to ./output/")
