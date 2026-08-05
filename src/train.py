import os
import json
import joblib
import pandas as pd
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier,
                              VotingClassifier, BaggingClassifier, StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.utils import MODELS_DIR, setup_logging, save_model, save_json
from src.data_loader import load_data
from src.feature_engineering import get_preprocessor

setup_logging()
logger = logging.getLogger(__name__)

def get_models():
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'ExtraTrees': ExtraTreesClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'NaiveBayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', verbosity=0, random_state=42),
        'LightGBM': LGBMClassifier(verbose=-1, random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }
    
    # Ensembles
    voting = VotingClassifier(
        estimators=[('rf', models['RandomForest']), ('xgb', models['XGBoost']), ('cb', models['CatBoost'])],
        voting='soft'
    )
    models['Voting'] = voting
    
    bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)
    models['Bagging'] = bagging
    
    stacking = StackingClassifier(
        estimators=[('rf', models['RandomForest']), ('xgb', models['XGBoost']), ('cb', models['CatBoost'])],
        final_estimator=LogisticRegression()
    )
    models['Stacking'] = stacking
    
    return models

def train_models(X, y):
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    best_score = 0
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            results[name] = mean_score
            logger.info(f"{name} CV Accuracy: {mean_score:.4f}")
            
            # Fit on full data
            model.fit(X, y)
            save_model(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
                best_model = model
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            
    save_json(results, os.path.join(MODELS_DIR, 'model_results.json'))
    
    if best_model:
        save_model(best_model, os.path.join(MODELS_DIR, 'best_model.joblib'))
        logger.info(f"Best model is {best_model_name} with accuracy {best_score:.4f}")
        
    return results, best_model_name, best_model

if __name__ == "__main__":
    df = load_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(X, y)
    
    feature_names = preprocessor.named_steps['col_trans'].get_feature_names_out()
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib'))
    
    train_models(X_processed, y)
