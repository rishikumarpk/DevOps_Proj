"""
Shared helpers: dataset schema and loading.

Why this file exists:
  train.py, predict.py and app.py all need to agree on the exact feature
  order. Keeping that in one place means the model you TRAIN and the model
  you SERVE handle data identically -- a very common source of bugs in ML
  systems otherwise.
"""
from pathlib import Path

import pandas as pd

# The 10 input features (mean-value measurements from the Wisconsin Breast
# Cancer Diagnostic dataset), in the order the model expects them. The
# FastAPI request body and any prediction call must use this same order.
FEATURES = [
    "Radius",
    "Texture",
    "Perimeter",
    "Area",
    "Smoothness",
    "Compactness",
    "Concavity",
    "ConcavePoints",
    "Symmetry",
    "FractalDimension",
]
TARGET = "Diagnosis"  # 1 = malignant, 0 = benign


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the breast cancer CSV and return a DataFrame with the expected columns."""
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def default_data_path() -> str:
    """data/breast_cancer.csv resolved relative to the project root."""
    return str(Path(__file__).resolve().parents[1] / "data" / "breast_cancer.csv")
