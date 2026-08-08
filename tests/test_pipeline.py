"""
Sanity checks for the pipeline and API. These are the checks CI runs.
"""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app import app  # noqa: E402
from src.predict import load_model, predict  # noqa: E402

client = TestClient(app)

# A realistic sample (first row of the dataset -> known malignant).
SAMPLE = {
    "Radius": 17.99,
    "Texture": 10.38,
    "Perimeter": 122.8,
    "Area": 1001.0,
    "Smoothness": 0.1184,
    "Compactness": 0.2776,
    "Concavity": 0.3001,
    "ConcavePoints": 0.1471,
    "Symmetry": 0.2419,
    "FractalDimension": 0.0787,
}


def test_model_loads():
    """The trained pipeline should load and expose predict/predict_proba."""
    model = load_model()
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_helper_shape():
    """predict() returns the expected keys and a probability in [0, 1]."""
    out = predict(SAMPLE)
    assert set(out) == {"prediction", "label", "probability"}
    assert out["prediction"] in (0, 1)
    assert out["label"] in ("malignant", "benign")
    assert 0.0 <= out["probability"] <= 1.0


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_endpoint_rejects_bad_input():
    """Missing a required field should be a 422 validation error."""
    bad = dict(SAMPLE)
    del bad["Radius"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
