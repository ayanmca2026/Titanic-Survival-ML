import time
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

try:
    from .schemas import (
        PassengerInput, PredictionOutput, BatchInput, 
        BatchOutput, HealthResponse, ModelInfoResponse
    )
    from .middleware import setup_middleware
except ImportError:
    from schemas import (
        PassengerInput, PredictionOutput, BatchInput, 
        BatchOutput, HealthResponse, ModelInfoResponse
    )
    from middleware import setup_middleware

# Setup predictions logger (safe for serverless environments)
try:
    log_dir = Path("/tmp/logs" if os.name != "nt" else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    pred_logger = logging.getLogger("predictions")
    pred_logger.setLevel(logging.INFO)
    pred_handler = logging.FileHandler(log_dir / "predictions.log")
    pred_handler.setFormatter(logging.Formatter('%(message)s'))
    pred_logger.addHandler(pred_handler)
except Exception:
    pred_logger = logging.getLogger("predictions")

# Globals
model = None
pipeline = None
feature_names = []
model_metrics = {}
START_TIME = time.time()
REQUEST_COUNTS = {"predict": 0, "predict_batch": 0}

def manual_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback manual preprocessing when pipeline is unavailable."""
    df = df.copy()
    
    # 1. Extract Title
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(
            ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
            'Rare'
        )
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        df['Title'] = df['Title'].fillna('Unknown')
    else:
        df['Title'] = 'Unknown'
        
    # 2. FamilySize and IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 4. Handle missing values
    df['Age'] = df['Age'].fillna(29.69911764705882) # Example mean
    df['Fare'] = df['Fare'].fillna(32.204207968574636) # Example mean
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # 5 & 6. Encode and Scale (simplified dummy encoding)
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Title'] = df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5, 'Unknown': 0}).fillna(0)
    
    # Select expected features (example)
    expected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    
    for f in expected_features:
        if f not in df.columns:
            df[f] = 0
            
    return df[expected_features]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, pipeline, feature_names, model_metrics
    
    model_dir = Path("models")
    model_path = model_dir / "best_model.joblib"
    pipeline_path = model_dir / "preprocessing_pipeline.joblib"
    features_path = model_dir / "feature_names.joblib"
    metrics_path = model_dir / "best_model_metrics.json"
    
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            logging.info(f"Loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            
    if pipeline_path.exists():
        try:
            pipeline = joblib.load(pipeline_path)
            logging.info(f"Loaded pipeline from {pipeline_path}")
        except Exception as e:
            logging.error(f"Failed to load pipeline: {e}")
            
    if features_path.exists():
        try:
            feature_names = joblib.load(features_path)
        except Exception as e:
            logging.error(f"Failed to load feature names: {e}")
            
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load metrics: {e}")
            
    yield
    # Cleanup on shutdown
    model = None
    pipeline = None

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting passenger survival on the Titanic",
    version="1.0.0",
    lifespan=lifespan
)

setup_middleware(app)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Titanic Survival Prediction API",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        uptime=time.time() - START_TIME,
        model_version="1.0",
        model_loaded=(model is not None)
    )

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    if not model:
        raise HTTPException(status_code=404, detail="Model not loaded")
        
    return ModelInfoResponse(
        model_name=model.__class__.__name__ if model else "Unknown",
        version="1.0",
        features=feature_names if feature_names else ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "FamilySize", "IsAlone"],
        metrics=model_metrics,
        training_date=model_metrics.get("training_date", "Unknown")
    )

def predict_single(passenger: PassengerInput) -> PredictionOutput:
    if not model:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")
        
    passenger_dict = passenger.model_dump()
    df = pd.DataFrame([passenger_dict])
    
    if pipeline:
        try:
            X = pipeline.transform(df)
        except Exception as e:
            logging.error(f"Pipeline transform error: {e}")
            X = manual_preprocessing(df)
    else:
        X = manual_preprocessing(df)
        
    # Predict
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][1])
        survived = prob >= 0.5
    else:
        survived = bool(model.predict(X)[0])
        prob = 1.0 if survived else 0.0
        
    output = PredictionOutput(
        survived=survived,
        survival_probability=prob,
        prediction_label="Survived" if survived else "Did Not Survive",
        passenger_summary=passenger_dict
    )
    
    # Log prediction
    log_entry = {
        "timestamp": time.time(),
        "input": passenger_dict,
        "prediction": survived,
        "probability": prob
    }
    pred_logger.info(json.dumps(log_entry))
    
    return output

@app.post("/predict", response_model=PredictionOutput)
async def predict(passenger: PassengerInput):
    REQUEST_COUNTS["predict"] += 1
    return predict_single(passenger)

@app.post("/predict_batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput):
    REQUEST_COUNTS["predict_batch"] += 1
    predictions = []
    survived_count = 0
    
    for passenger in batch.passengers:
        pred = predict_single(passenger)
        predictions.append(pred)
        if pred.survived:
            survived_count += 1
            
    summary = {
        "total": len(batch.passengers),
        "survived_count": survived_count,
        "did_not_survive_count": len(batch.passengers) - survived_count
    }
    
    return BatchOutput(predictions=predictions, summary=summary)
