import os
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
import optuna
from src.utils import MODELS_DIR, setup_logging, save_model
from src.train import get_models
import joblib

setup_logging()
logger = logging.getLogger(__name__)

def tune_models(X, y):
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tuned_models = {}
    
    # GridSearchCV for LogReg
    if 'LogisticRegression' in models:
        logger.info("Tuning LogisticRegression...")
        param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        gs = GridSearchCV(models['LogisticRegression'], param_grid, cv=cv, scoring='accuracy')
        gs.fit(X, y)
        tuned_models['LogisticRegression'] = gs.best_estimator_
        save_model(gs.best_estimator_, os.path.join(MODELS_DIR, 'LogisticRegression_tuned.joblib'))
        
    # RandomizedSearchCV for RF
    if 'RandomForest' in models:
        logger.info("Tuning RandomForest...")
        param_dist = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
        rs = RandomizedSearchCV(models['RandomForest'], param_dist, cv=cv, scoring='accuracy', n_iter=5, random_state=42)
        rs.fit(X, y)
        tuned_models['RandomForest'] = rs.best_estimator_
        save_model(rs.best_estimator_, os.path.join(MODELS_DIR, 'RandomForest_tuned.joblib'))
        
    # Optuna for XGBoost (Simplified for speed)
    logger.info("Tuning XGBoost with Optuna...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        clf = XGBClassifier(**params, random_state=42)
        return cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    
    from xgboost import XGBClassifier
    best_xgb = XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
    best_xgb.fit(X, y)
    tuned_models['XGBoost'] = best_xgb
    save_model(best_xgb, os.path.join(MODELS_DIR, 'XGBoost_tuned.joblib'))
    
    logger.info("Tuning complete.")
    return tuned_models
