"""
Prediction helper shared by the API and the tests.

Loads the saved best model (models/best_model.joblib) once and turns a dict of
raw feature values into a prediction + probability. Keeping this logic here (not
inside app.py) means the tests can exercise it without spinning up a web server.
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline once and cache it in memory."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


def predict(features: dict) -> dict:
    """Predict malignancy from a dict of the 10 feature values.

    Returns the class (0/1), a human label, and P(malignant). The pipeline
    does all imputation/scaling internally, so raw values go straight in.
    """
    model = load_model()
    # Order the incoming values exactly as the model expects.
    row = pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": pred,
        "label": "malignant" if pred == 1 else "benign",
        "probability": round(proba, 4),
    }
