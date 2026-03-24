from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# =========================
# Load Saved Objects
# =========================
with open("cotton_ann_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("stage_encoder.pkl", "rb") as f:
    stage_encoder = pickle.load(f)

# =========================
# FastAPI App
# =========================
app = FastAPI(title="Cotton Disease Prediction API")

# =========================
# Input Schema
# =========================
class InputData(BaseModel):
    growth_stage: str
    vegetation_stress_score: float
    water_stress_score: float
    soil_stress_score: float
    final_stress_percent: float
    gdd_min: float
    gdd_max: float

# =========================
# Disease Labels
# =========================
disease_labels = [
    "damping_off",
    "root_rot",
    "bacterial_blight",
    "alternaria_leaf_spot",
    "fusarium_wilt",
    "verticillium_wilt",
    "boll_rot"
]

# =========================
# Health Check
# =========================
@app.get("/")
def home():
    return {"message": "Cotton Disease ANN API is running"}

# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
def predict(data: InputData):
    try:
        # Encode stage
        stage_encoded = stage_encoder.transform([data.growth_stage])[0]

        # Prepare input array
        input_features = np.array([[
            data.vegetation_stress_score,
            data.water_stress_score,
            data.soil_stress_score,
            data.final_stress_percent,
            data.gdd_min,
            data.gdd_max,
            stage_encoded
        ]])

        # Scale
        input_scaled = scaler.transform(input_features)

        # Predict
        predictions = model.predict(input_scaled)[0]

        # Map output
        result = {
            disease_labels[i]: float(round(predictions[i], 4))
            for i in range(len(disease_labels))
        }

        return {
            "status": "success",
            "predictions": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }