from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from pathlib import Path
import os

app = FastAPI(title="Credit Risk Prediction API")

# Load model from MLflow
MODEL_PATH = "models:/CreditRiskModel/Production"
model = mlflow.pyfunc.load_model(MODEL_PATH)

class PredictionInput(BaseModel):
    recency: float
    frequency: float
    monetary: float
    # Add other features matching your training data

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Make predictions with versioned model"""
    input_df = pd.DataFrame([input_data.dict()])
    proba = float(model.predict(input_df)[0])
    return {
        "risk_probability": proba,
        "risk_class": int(proba >= 0.5)  # Threshold at 0.5
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": MODEL_PATH}