"""
Training stage.

Trains THREE classifiers on the Wisconsin Breast Cancer data, evaluates each
with five metrics, logs everything to MLflow, prints a comparison table, then:
  - registers the best model (by ROC-AUC) in the MLflow Model Registry, and
  - saves that same model to models/best_model.joblib for the API/Docker to use.

Run:  python src/train.py
"""
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make `import utils` work whether this file is run as `python src/train.py`
# or imported as part of the `src` package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FEATURES, TARGET, default_data_path, load_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
# MLflow 3.x deprecated the plain-file tracking store, so we use a local SQLite
# database as the backend. It's still fully local (a single mlflow.db file, no
# server needed) and it's what enables the Model Registry to work.
MLFLOW_DB = ROOT / "mlflow.db"

# The name the best model gets in the MLflow Model Registry.
REGISTERED_MODEL_NAME = "breast-cancer-classifier"
EXPERIMENT_NAME = "breast-cancer-classification"


def build_pipeline(classifier, scale: bool) -> Pipeline:
    """Wrap preprocessing + a classifier into one object.

    Baking preprocessing INTO the model means the API can feed raw feature
    values straight in -- the pipeline imputes/scales automatically, the
    exact same way it did during training.

    Steps:
      1. impute median  (fill any missing values defensively)
      2. scale          (only for Logistic Regression, which is sensitive to
                         feature scale; tree models don't need it)
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def model_configs():
    """The three models we compare, with the hyperparameters we log to MLflow.

    - Logistic Regression: simple, fast, interpretable linear baseline.
    - Random Forest: bagged trees, robust, handles non-linearities, low tuning.
    - Gradient Boosting: sequentially boosted trees, usually the strongest of
      the three on tabular data (used instead of XGBoost to avoid an extra
      native dependency -- it's pure scikit-learn and Dockerises cleanly).
    """
    return {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, C=1.0),
            "scale": True,
            "params": {"max_iter": 1000, "C": 1.0},
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=300, max_depth=6, random_state=42
            ),
            "scale": False,
            "params": {"n_estimators": 300, "max_depth": 6, "random_state": 42},
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
            ),
            "scale": False,
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
            },
        },
    }


def evaluate(model, X_test, y_test) -> dict:
    """Compute the five classification metrics the brief asks for."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]  # P(malignant) for ROC-AUC
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # Local SQLite MLflow tracking store (backend = mlflow.db, artifacts in
    # ./mlartifacts). To browse it later:
    #   mlflow ui --backend-store-uri sqlite:///mlflow.db
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(default_data_path())
    X = df[FEATURES]
    y = df[TARGET]

    # Stratify keeps the malignant/benign ratio the same in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    model_uris = {}

    for name, cfg in model_configs().items():
        with mlflow.start_run(run_name=name):
            pipe = build_pipeline(cfg["estimator"], cfg["scale"])
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_params(cfg["params"])
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(
                pipe, name="model", serialization_format="cloudpickle"
            )

            model_uris[name] = info.model_uri
            results.append({"model": name, **metrics})
            print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # ---- Comparison table ----
    table = pd.DataFrame(results).set_index("model").sort_values("roc_auc", ascending=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(table.round(4).to_string())

    table.to_csv(MODELS_DIR / "metrics_comparison.csv")
    with open(MODELS_DIR / "metrics_comparison.json", "w") as f:
        json.dump(table.reset_index().to_dict(orient="records"), f, indent=2)

    # ---- Pick the best model (highest ROC-AUC) ----
    best_name = table.index[0]
    best_metrics = table.loc[best_name].to_dict()
    model_uri = model_uris[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC={best_metrics['roc_auc']:.4f})")

    # ---- Register the best model in the MLflow Model Registry ----
    registered_version = None
    try:
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        registered_version = mv.version
        # Modern MLflow uses named aliases instead of stages; mark this version
        # as the one serving "production". Wrapped in try/except so an older
        # MLflow that lacks aliases still doesn't crash the whole run.
        try:
            from mlflow.tracking import MlflowClient

            MlflowClient().set_registered_model_alias(
                REGISTERED_MODEL_NAME, "production", registered_version
            )
        except Exception as e:  # pragma: no cover - version-dependent
            print(f"(alias not set, non-fatal: {e})")
        print(
            f"Registered '{REGISTERED_MODEL_NAME}' version {registered_version} "
            f"and aliased it as 'production'."
        )
    except Exception as e:  # pragma: no cover - registry optional
        print(f"(Model registry step skipped, non-fatal: {e})")

    # ---- Save the best model + metadata for the API/Docker ----
    # The API loads THIS file rather than the MLflow registry: it's a single
    # self-contained artifact, so the Docker image needs no mlruns/ store.
    best_pipe = mlflow.sklearn.load_model(model_uri)
    joblib.dump(best_pipe, MODELS_DIR / "best_model.joblib")

    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "metrics": {k: float(v) for k, v in best_metrics.items()},
        "registered_model": REGISTERED_MODEL_NAME,
        "registered_version": registered_version,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved models/best_model.joblib and models/model_metadata.json")


if __name__ == "__main__":
    main()
