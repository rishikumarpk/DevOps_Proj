"""
Shared test setup.

If the model artifact doesn't exist yet (e.g. a fresh checkout in CI), train it
once before the tests run so every test has a model to load. Training is quick
on this small dataset.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def ensure_model_trained():
    model_path = ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        from src import train

        train.main()
    return model_path
