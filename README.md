# Breast Cancer Diagnosis MLOps Project

An end-to-end MLOps pipeline for **Breast Cancer Diagnosis** (Malignant vs. Benign) using scikit-learn, MLflow, DVC, FastAPI, Docker, and GitHub Actions. Built on the Wisconsin Breast Cancer Diagnostic dataset (scikit-learn's built-in `load_breast_cancer`, reduced to the 10 "mean" measurement features).

This project mirrors the structure of a diabetes-prediction MLOps capstone, applied to a different classification problem.

## 🏗️ Project Structure

```
.
├── data/                        # Dataset (versioned with DVC)
│   ├── breast_cancer.csv
│   └── breast_cancer.csv.dvc    # DVC pointer (tracked by git)
├── models/                      # Trained model artifacts (best_model.joblib + metrics)
├── src/
│   ├── train.py                 # Trains 3 models + MLflow tracking + registry
│   ├── app.py                   # FastAPI prediction service
│   ├── predict.py               # Inference helper (loads the best model)
│   └── utils.py                 # Data loading utilities
├── tests/
│   ├── conftest.py              # Trains a model if none exists (CI safety)
│   └── test_pipeline.py         # Model + API endpoint tests
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI pipeline
├── Dockerfile                   # Container image (app + deps + model)
├── requirements.txt
├── dvc.yaml                     # DVC pipeline definition
├── dvc.lock                     # Reproducibility lock file
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dataset & DVC tracking
The dataset is already tracked with DVC (`data/breast_cancer.csv.dvc`). To inspect the versioning:
```bash
dvc dag        # shows: data/breast_cancer.csv.dvc -> train
dvc status     # "Data and pipelines are up to date."
```

### 3. Train models
```bash
python src/train.py
```
This trains **Logistic Regression, Random Forest, and Gradient Boosting** — logs every experiment to MLflow, prints a comparison table, and registers the best model (by ROC-AUC) in the MLflow Model Registry.

### 4. View MLflow experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://localhost:5000 in your browser to compare the three runs and see the registered model.

### 5. Run the FastAPI server
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/docs for the interactive Swagger UI.

### 6. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Radius": 17.99,
    "Texture": 10.38,
    "Perimeter": 122.8,
    "Area": 1001.0,
    "Smoothness": 0.1184,
    "Compactness": 0.2776,
    "Concavity": 0.3001,
    "ConcavePoints": 0.1471,
    "Symmetry": 0.2419,
    "FractalDimension": 0.0787
  }'
```
Response:
```json
{ "prediction": 1, "label": "malignant", "probability": 0.97 }
```

### 7. Run tests
```bash
pytest tests/ -v
```

### 8. Build & run Docker
```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

## 📊 Models Trained

Full metrics on a stratified 20% hold-out test set (this run):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| **RandomForest** ⭐ | 0.939 | 0.927 | 0.905 | 0.916 | **0.985** |
| LogisticRegression | 0.930 | 0.886 | 0.929 | 0.907 | 0.984 |
| GradientBoosting | 0.947 | 0.950 | 0.905 | 0.927 | 0.981 |

The best-performing model (by ROC-AUC) is automatically registered in the MLflow Model Registry as **`breast-cancer-classifier`** and marked with the **`production`** alias. All three models bake preprocessing into an sklearn `Pipeline` (median imputation, and — for Logistic Regression — standard scaling), so raw feature values can be sent straight to the API.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + which model is loaded |
| POST | `/predict` | Single prediction (returns class, label, probability) |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |

## 🔄 CI/CD Pipeline (GitHub Actions)

The workflow in `.github/workflows/ci.yml` runs on every push/PR:

1. **Checkout** the repository
2. **Set up Python** 3.12
3. **Install** dependencies
4. **Train** the models (produces the model artifact)
5. **Run** `pytest tests/`
6. **Build** the Docker image and smoke-test `/health` in the container

Build + test only — no deployment step.

## 📋 Dataset

**Wisconsin Breast Cancer Diagnostic Dataset** (via `sklearn.datasets.load_breast_cancer`, reduced to the 10 mean-value features for a simpler, more approachable feature set):

- 569 samples, 10 numerical features
- Binary classification: `1` = Malignant, `0` = Benign (`Diagnosis` column)
- Features: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, ConcavePoints, Symmetry, FractalDimension
- Versioned with DVC

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| ML | scikit-learn |
| Experiment Tracking | MLflow (SQLite backend) |
| API | FastAPI + Uvicorn |
| Data Versioning | DVC |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |

## 📝 Notes on adapting this template

This project was generated as a sibling to a diabetes-prediction MLOps capstone, keeping the same architecture (train → track → register → serve → containerize → CI) but swapping in a different dataset and domain. To adapt it further to your own dataset:

1. Replace `data/breast_cancer.csv` and update `src/utils.py` (`FEATURES`, `TARGET`).
2. Adjust `src/app.py`'s Pydantic model to match your new feature names.
3. Re-run `python src/train.py` to retrain, then `pytest tests/ -v` to confirm everything still passes.
4. If any raw feature values legitimately can't be zero (like `Glucose` in the original diabetes project), add back a `zero_to_nan`-style step in `build_pipeline` before imputation.
