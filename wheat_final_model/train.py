"""
train_ann.py

Purpose:
Train an Artificial Neural Network (ANN) to map:
API-derived stress + growth-stage context → disease probabilities

Outputs:
- disease_probability_ann.pkl
- stress_scaler.pkl
- crop_encoder.pkl
- stage_encoder.pkl
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------
# 1. Load dataset (API-prepared training data)
# -------------------------------------------------
DATA_PATH = "wheat_str_final.xlsx"
df = pd.read_excel(DATA_PATH)

# -------------------------------------------------
# 2. Encode categorical metadata (NOT learned)
# -------------------------------------------------
crop_encoder = LabelEncoder()
stage_encoder = LabelEncoder()

df["crop_enc"] = crop_encoder.fit_transform(df["crop"])
df["stage_enc"] = stage_encoder.fit_transform(df["growth_stage"])

# -------------------------------------------------
# 3. FINAL ANN INPUT CONTRACT (API outputs only)
# -------------------------------------------------
FEATURE_COLUMNS = [
    "vegetation_stress_score",
    "water_stress_score",
    "soil_stress_score",
    "final_stress_percent",
    "gdd_min",
    "gdd_max",
    "crop_enc",
    "stage_enc"
]

TARGET_COLUMNS = [
    "prob_rust",
    "prob_blight",
    "prob_root_rot",
    "prob_healthy"
]

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMNS]

# -------------------------------------------------
# 4. Train / test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# 5. Scale numerical inputs
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 6. ANN model
# -------------------------------------------------
ann_model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1200,
    random_state=42
)

ann_model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# 7. Evaluation (sanity check)
# -------------------------------------------------
y_pred = ann_model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")

# -------------------------------------------------
# 8. Save production artifacts
# -------------------------------------------------
with open("disease_probability_ann.pkl", "wb") as f:
    pickle.dump(ann_model, f)

with open("stress_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("crop_encoder.pkl", "wb") as f:
    pickle.dump(crop_encoder, f)

with open("stage_encoder.pkl", "wb") as f:
    pickle.dump(stage_encoder, f)

print(" Training complete. Production files saved.")
