"""Titanic Survival Prediction — CLI Entry Point.

Usage:
    python run.py --mode full        # Run entire pipeline
    python run.py --mode train       # Train models only
    python run.py --mode evaluate    # Evaluate best model
    python run.py --mode api         # Start FastAPI server
    python run.py --mode dashboard   # Start Streamlit dashboard
    python run.py --mode predict     # Interactive prediction
"""

# Set non-interactive matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import setup_logging, MODELS_DIR, IMAGES_DIR, REPORTS_DIR
from src.data_loader import load_data, dataset_overview
from src.feature_engineering import get_preprocessor
from src.train import train_models
from src.evaluate import evaluate_model
from src.predict import predict_single
from src.explain import explain_model
from src.feature_selection import run_feature_selection

setup_logging()
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Execute the complete ML pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("STARTING FULL TITANIC ML PIPELINE")
    logger.info("=" * 60)

    # Phase 1: Data Loading
    logger.info("\n📥 Phase 1: Loading Dataset...")
    df = load_data()
    overview = dataset_overview(df)
    logger.info(f"Dataset shape: {df.shape}")

    # Phase 2: Preprocessing + Feature Engineering
    logger.info("\n🔧 Phase 2-3: Preprocessing & Feature Engineering...")
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Train/test split BEFORE fitting pipeline (to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = list(preprocessor.named_steps['col_trans'].get_feature_names_out())
    logger.info(f"Processed features ({len(feature_names)}): {feature_names}")

    # Save pipeline and feature names
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
    joblib.dump(y_test, os.path.join(MODELS_DIR, 'y_test.joblib'))
    logger.info("Pipeline and feature names saved.")

    # Save processed data
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    pd.DataFrame(X_train_proc, columns=feature_names).to_csv(
        os.path.join(processed_dir, 'X_train.csv'), index=False
    )
    pd.DataFrame(X_test_proc, columns=feature_names).to_csv(
        os.path.join(processed_dir, 'X_test.csv'), index=False
    )

    # Phase 4: Feature Selection Analysis
    logger.info("\n🎯 Phase 4: Feature Selection...")
    try:
        run_feature_selection(X_train_proc, y_train, feature_names)
    except Exception as e:
        logger.warning(f"Feature selection skipped: {e}")

    # Phase 5: Model Training
    logger.info("\n🤖 Phase 5: Training Models...")
    results, best_name, best_model = train_models(X_train_proc, y_train)
    logger.info(f"\n🏆 Best Model: {best_name}")
    logger.info(f"Results: {json.dumps(results, indent=2)}")

    # Phase 6: Hyperparameter Tuning (on best model type)
    logger.info("\n⚙️ Phase 6: Hyperparameter Tuning...")
    try:
        from src.hyperparameter_tuning import tune_models
        tuned_results = tune_models(X_train_proc, y_train)
        logger.info(f"Tuning results: {json.dumps({k: v for k, v in tuned_results.items() if isinstance(v, (int, float, str))}, indent=2, default=str)}")
    except Exception as e:
        logger.warning(f"Hyperparameter tuning skipped: {e}")

    # Phase 7: Model Evaluation
    logger.info("\n📊 Phase 7: Model Evaluation...")
    best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
    metrics = evaluate_model(best_model, X_test_proc, y_test, model_name=best_name)
    logger.info(f"Test Metrics: {json.dumps(metrics, indent=2)}")

    # Phase 8: Explainable AI
    logger.info("\n🔍 Phase 8: Generating Explanations...")
    try:
        explain_model(X_test_proc, feature_names, model=best_model)
    except Exception as e:
        logger.warning(f"Explainability skipped: {e}")

    # Generate EDA visualizations
    logger.info("\n📈 Generating EDA Visualizations...")
    try:
        generate_eda_plots(df)
    except Exception as e:
        logger.warning(f"EDA plots skipped: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info(f"   Models saved to: {MODELS_DIR}")
    logger.info(f"   Images saved to: {IMAGES_DIR}")
    logger.info(f"   Reports saved to: {REPORTS_DIR}")
    logger.info("=" * 60)
    
    return results, metrics


def generate_eda_plots(df):
    """Generate static EDA plots for the images/ directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Survival Count
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Survived', palette=['#FF4B4B', '#00FF88'], ax=ax)
    ax.set_xticklabels(['Did Not Survive', 'Survived'])
    ax.set_title('Survival Count', fontsize=14, fontweight='bold')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'survival_count.png'), dpi=150)
    plt.close()

    # 2. Gender vs Survival
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Sex', hue='Survived', palette=['#FF4B4B', '#00FF88'], ax=ax)
    ax.set_title('Gender vs Survival', fontsize=14, fontweight='bold')
    ax.legend(['Did Not Survive', 'Survived'])
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'gender_vs_survival.png'), dpi=150)
    plt.close()

    # 3. Pclass vs Survival
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Pclass', hue='Survived', palette=['#FF4B4B', '#00FF88'], ax=ax)
    ax.set_title('Passenger Class vs Survival', fontsize=14, fontweight='bold')
    ax.legend(['Did Not Survive', 'Survived'])
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'pclass_vs_survival.png'), dpi=150)
    plt.close()

    # 4. Age Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Age'].dropna().hist(bins=30, ax=ax, color='#00D4FF', edgecolor='white', alpha=0.7)
    ax.set_title('Age Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'age_distribution.png'), dpi=150)
    plt.close()

    # 5. Fare Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df['Fare'].dropna().hist(bins=30, ax=ax, color='#FFB800', edgecolor='white', alpha=0.7)
    ax.set_title('Fare Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fare')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'fare_distribution.png'), dpi=150)
    plt.close()

    # 6. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # 7. Violin Plot - Age by Pclass
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x='Pclass', y='Age', hue='Survived',
                   split=True, palette=['#FF4B4B', '#00FF88'], ax=ax)
    ax.set_title('Age Distribution by Pclass & Survival', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'violin_age_pclass.png'), dpi=150)
    plt.close()

    # 8. Box Plot - Fare by Pclass
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='Pclass', y='Fare', palette=['#636EFA', '#00D4FF', '#FFB800'], ax=ax)
    ax.set_title('Fare by Passenger Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_fare_pclass.png'), dpi=150)
    plt.close()

    logger.info("EDA plots generated successfully.")


def run_train():
    """Train models only."""
    df = load_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)

    feature_names = list(preprocessor.named_steps['col_trans'].get_feature_names_out())
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
    joblib.dump(y_test, os.path.join(MODELS_DIR, 'y_test.joblib'))

    results, best_name, best_model = train_models(X_train_proc, y_train)
    logger.info(f"Training complete. Best: {best_name}")
    return results


def run_evaluate():
    """Evaluate the best saved model."""
    df = load_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib'))
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_proc = preprocessor.transform(X_test)

    metrics = evaluate_model(model, X_test_proc, y_test, model_name="BestModel")
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def run_api():
    """Start the FastAPI server."""
    import uvicorn
    logger.info("Starting FastAPI server on http://localhost:8000")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)


def run_dashboard():
    """Start the Streamlit dashboard."""
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'streamlit_app.py')
    logger.info(f"Starting Streamlit dashboard from {dashboard_path}")
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.port', '8501',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ])


def run_predict_interactive():
    """Interactive CLI prediction."""
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessing_pipeline.joblib'))
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))

    print("\n🚢 Titanic Survival Predictor")
    print("=" * 40)

    name = input("Passenger Name (e.g., Mr. John Smith): ") or "Mr. Unknown"
    pclass = int(input("Ticket Class (1/2/3): ") or "3")
    sex = input("Sex (male/female): ") or "male"
    age = float(input("Age: ") or "30")
    sibsp = int(input("Siblings/Spouses aboard: ") or "0")
    parch = int(input("Parents/Children aboard: ") or "0")
    fare = float(input("Fare paid: ") or "32.0")
    embarked = input("Port (S/C/Q): ") or "S"

    passenger = pd.DataFrame([{
        'PassengerId': 0, 'Pclass': pclass, 'Name': name, 'Sex': sex,
        'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Ticket': 'XXXXX',
        'Fare': fare, 'Cabin': 'U', 'Embarked': embarked
    }])

    X_proc = preprocessor.transform(passenger)
    prob = model.predict_proba(X_proc)[0][1]
    survived = prob >= 0.5

    print("\n" + "=" * 40)
    print(f"{'🟢 SURVIVED' if survived else '🔴 DID NOT SURVIVE'}")
    print(f"Survival Probability: {prob:.1%}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="🚢 Titanic Survival Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode full        Run entire pipeline
  python run.py --mode train       Train models only
  python run.py --mode evaluate    Evaluate best model
  python run.py --mode api         Start FastAPI server
  python run.py --mode dashboard   Start Streamlit dashboard
  python run.py --mode predict     Interactive prediction
        """
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'train', 'evaluate', 'predict', 'api', 'dashboard'],
        default='full',
        help='Pipeline mode to run'
    )
    args = parser.parse_args()

    if args.mode == 'full':
        run_full_pipeline()
    elif args.mode == 'train':
        run_train()
    elif args.mode == 'evaluate':
        run_evaluate()
    elif args.mode == 'predict':
        run_predict_interactive()
    elif args.mode == 'api':
        run_api()
    elif args.mode == 'dashboard':
        run_dashboard()


if __name__ == "__main__":
    main()
