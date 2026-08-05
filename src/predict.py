import os
import joblib
import pandas as pd
from src.utils import MODELS_DIR

def load_pipeline_and_model():
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib'))
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    return preprocessor, model

def predict_single(passenger_dict):
    df = pd.DataFrame([passenger_dict])
    preprocessor, model = load_pipeline_and_model()
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None
    return {'prediction': int(pred), 'probability': float(proba) if proba is not None else None}

def predict_batch(df):
    preprocessor, model = load_pipeline_and_model()
    X = preprocessor.transform(df)
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else [None]*len(preds)
    return preds, probas
