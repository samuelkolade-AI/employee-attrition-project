"""
FastAPI application for Employee Attrition Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from datetime import datetime

from predict import AttritionPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict employee attrition risk using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
try:
    predictor = AttritionPredictor()
except Exception as e:
    print(f"⚠️  Warning: Could not load model - {e}")
    predictor = None

# Pydantic models


class EmployeeFeatures(BaseModel):
    """Employee features for prediction."""
    age: int = Field(..., ge=18, le=70)
    tenure_years: int = Field(..., ge=0)
    monthly_income: float = Field(..., gt=0)
    satisfaction_score: float = Field(..., ge=1, le=4)
    overtime_flag: int = Field(..., ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "tenure_years": 5,
                "monthly_income": 5000,
                "satisfaction_score": 2.5,
                "overtime_flag": 1
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    attrition_probability: float
    prediction: int
    risk_level: str
    top_risk_factors: List[str]
    timestamp: str

# API Endpoints


@app.get("/")
def read_root():
    """Root endpoint with API info."""
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict attrition for single employee",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_attrition(features: EmployeeFeatures):
    """
    Predict attrition risk for a single employee.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features_dict = features.dict()
        result = predictor.predict_single(features_dict)
        result['timestamp'] = datetime.now().isoformat()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
