from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# -------------------------
# Load production artifacts
# -------------------------
with open("disease_probability_ann.pkl", "rb") as f:
    model = pickle.load(f)

with open("stress_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("crop_encoder.pkl", "rb") as f:
    crop_encoder = pickle.load(f)

with open("stage_encoder.pkl", "rb") as f:
    stage_encoder = pickle.load(f)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="Crop Disease Probability API",
    description="Stress → Disease Probability ANN (Production)",
    version="1.0.0"
)

# -------------------------
# Input schema (API OUTPUTS ONLY)
# -------------------------
class DiseaseInput(BaseModel):
    crop: str
    growth_stage: str

    vegetation_stress_score: float
    water_stress_score: float
    soil_stress_score: float
    final_stress_percent: float

    gdd_min: float
    gdd_max: float


# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict-disease")
def predict_disease(data: DiseaseInput):

    try:
        # Encode categorical metadata
        crop_enc = crop_encoder.transform([data.crop])[0]
        stage_enc = stage_encoder.transform([data.growth_stage])[0]

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Encoding error: {str(e)}"
        )

    # ANN input vector (STRICT CONTRACT)
    X = np.array([[
        data.vegetation_stress_score,
        data.water_stress_score,
        data.soil_stress_score,
        data.final_stress_percent,
        data.gdd_min,
        data.gdd_max,
        crop_enc,
        stage_enc
    ]])

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)[0]

    # Response
    return {
        "crop": data.crop,
        "growth_stage": data.growth_stage,
        "disease_probabilities": {
            "rust": round(float(preds[0]), 4),
            "blight": round(float(preds[1]), 4),
            "root_rot": round(float(preds[2]), 4),
            "healthy": round(float(preds[3]), 4)
        }
    }
