"""
Train logistic-regression model for heart-disease risk and save to ./output/
"""

import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ------------------------------------------------------------------
# 1. Config & paths
# ------------------------------------------------------------------
RANDOM_STATE = 40
CSV_PATH     = Path("Heart Prediction Quantum Dataset.csv")
OUT_DIR      = Path("output")
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH   = OUT_DIR / "logreg_pipeline.pkl"

# ------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
target = "HeartDisease" if "HeartDisease" in df.columns else df.columns[-1]
y = df[target]
X = df.drop(columns=[target])

numeric_features = X.columns.tolist()        # all columns numeric here

# ------------------------------------------------------------------
# 3. Build pipeline (scaler + logistic regression)
# ------------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[("scale", StandardScaler(), numeric_features)],
    remainder="passthrough"          # nothing else to process
)

pipe = Pipeline([
    ("prep", preprocess),
    ("clf" , LogisticRegression(max_iter=1000,
                                solver="liblinear",
                                random_state=RANDOM_STATE))
])

# ------------------------------------------------------------------
# 4. Fit on FULL dataset (tiny sample → use all rows)
# ------------------------------------------------------------------
pipe.fit(X, y)

# ------------------------------------------------------------------
# 5. Persist model
# ------------------------------------------------------------------
joblib.dump(pipe, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH.resolve()}")
