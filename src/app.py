"""
FastAPI prediction service.

Endpoints:
  GET  /health   -> liveness check + which model is loaded
  POST /predict  -> send the 10 feature values as JSON, get a diagnosis prediction

Run locally:  uvicorn src.app:app --reload
Then open the interactive docs at http://127.0.0.1:8000/docs
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import load_metadata, load_model, predict  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request isn't slow (and so
    the container fails fast if the model file is missing)."""
    load_model()
    yield


app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description="Predicts malignant vs. benign from the Wisconsin Breast Cancer feature set.",
    version="1.0.0",
    lifespan=lifespan,
)


class TumorFeatures(BaseModel):
    """Request body. Field names match the dataset columns exactly."""

    Radius: float = Field(..., examples=[17.99])
    Texture: float = Field(..., examples=[10.38])
    Perimeter: float = Field(..., examples=[122.8])
    Area: float = Field(..., examples=[1001.0])
    Smoothness: float = Field(..., examples=[0.1184])
    Compactness: float = Field(..., examples=[0.2776])
    Concavity: float = Field(..., examples=[0.3001])
    ConcavePoints: float = Field(..., examples=[0.1471])
    Symmetry: float = Field(..., examples=[0.2419])
    FractalDimension: float = Field(..., examples=[0.0787])


@app.get("/health")
def health():
    """Simple liveness probe used by tests, Docker, and CI."""
    meta = load_metadata()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": meta.get("model_name", "unknown"),
    }


@app.post("/predict")
def make_prediction(features: TumorFeatures):
    """Return {prediction, label, probability} for one sample."""
    return predict(features.model_dump())
