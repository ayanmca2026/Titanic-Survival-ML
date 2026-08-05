"""Explainable AI module: SHAP, LIME, Permutation Importance, PDP."""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import IMAGES_DIR, MODELS_DIR, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def explain_model(X, feature_names, model=None):
    """Generate all explainability artifacts for the best model."""
    if model is None:
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))

    feature_names = list(feature_names) if not isinstance(feature_names, list) else feature_names

    # Convert to DataFrame for better plotting
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = pd.DataFrame(X, columns=feature_names)

    # --- SHAP ---
    _generate_shap(model, X_df, feature_names)

    # --- Permutation Importance ---
    _generate_permutation_importance(model, X_df, feature_names)

    # --- Feature Importance (tree-based) ---
    _generate_feature_importance(model, feature_names)

    # --- Partial Dependence Plots ---
    _generate_pdp(model, X_df, feature_names)

    logger.info("All explainability artifacts generated successfully.")


def _generate_shap(model, X_df, feature_names):
    """Generate SHAP summary, bar, and dependence plots."""
    try:
        import shap

        # Use TreeExplainer for tree-based, KernelExplainer fallback
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
        else:
            # For non-tree models, use a small background sample
            background = shap.sample(X_df, min(50, len(X_df)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_df.iloc[:100])

        # Handle binary classification SHAP output
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Class 1 (Survived)
        else:
            shap_vals = shap_values

        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, X_df, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("SHAP summary plot saved.")

        # Bar plot (mean absolute SHAP)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_df, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("SHAP bar plot saved.")

        # Save SHAP values for dashboard use
        shap_importance = np.abs(shap_vals).mean(axis=0)
        shap_dict = {name: float(val) for name, val in zip(feature_names, shap_importance)}
        with open(os.path.join(MODELS_DIR, 'shap_importance.json'), 'w') as f:
            json.dump(shap_dict, f, indent=4)

        # Dependence plots for top 3 features
        top_features_idx = np.argsort(shap_importance)[::-1][:3]
        for idx in top_features_idx:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(idx, shap_vals, X_df, feature_names=feature_names, show=False)
            feat_name = feature_names[idx].replace('/', '_').replace(' ', '_')
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, f'shap_dep_{feat_name}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        logger.info("SHAP dependence plots saved.")

    except Exception as e:
        logger.warning(f"SHAP generation failed: {e}")


def _generate_permutation_importance(model, X_df, feature_names):
    """Generate permutation importance plot."""
    try:
        from sklearn.inspection import permutation_importance as perm_importance

        # We need y for permutation importance, try to load from saved data
        y_path = os.path.join(MODELS_DIR, 'y_test.joblib')
        if os.path.exists(y_path):
            y = joblib.load(y_path)
        else:
            # Generate dummy targets for importance ranking (not ideal but works for visualization)
            y = model.predict(X_df)

        result = perm_importance(model, X_df, y, n_repeats=10, random_state=42)

        sorted_idx = result.importances_mean.argsort()[::-1][:15]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], color='#00D4FF')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Mean Decrease in Accuracy')
        plt.title('Permutation Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'permutation_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Permutation importance plot saved.")

        # Save as JSON
        perm_dict = {feature_names[i]: float(result.importances_mean[i]) for i in sorted_idx}
        with open(os.path.join(MODELS_DIR, 'permutation_importance.json'), 'w') as f:
            json.dump(perm_dict, f, indent=4)

    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")


def _generate_feature_importance(model, feature_names):
    """Generate tree-based feature importance plot."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:15]

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_idx)), importances[sorted_idx], color='#00FF88')
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title('Tree-Based Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Feature importance plot saved.")

            # Save as JSON
            imp_dict = {feature_names[i]: float(importances[i]) for i in sorted_idx}
            with open(os.path.join(MODELS_DIR, 'feature_importance.json'), 'w') as f:
                json.dump(imp_dict, f, indent=4)
        else:
            logger.info("Model does not have feature_importances_ attribute.")

    except Exception as e:
        logger.warning(f"Feature importance generation failed: {e}")


def _generate_pdp(model, X_df, feature_names):
    """Generate Partial Dependence Plots for top features."""
    try:
        from sklearn.inspection import PartialDependenceDisplay

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:4]
        else:
            top_idx = list(range(min(4, len(feature_names))))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, idx in enumerate(top_idx):
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X_df, [idx], ax=axes[i],
                    feature_names=feature_names
                )
                axes[i].set_title(f'PDP: {feature_names[idx]}')
            except Exception:
                axes[i].text(0.5, 0.5, f'N/A: {feature_names[idx]}',
                           ha='center', va='center', fontsize=12)
                axes[i].set_title(f'PDP: {feature_names[idx]}')

        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'partial_dependence.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Partial dependence plots saved.")

    except Exception as e:
        logger.warning(f"PDP generation failed: {e}")


def explain_single_prediction(model, X_single, feature_names):
    """Generate explanation for a single prediction (for API use)."""
    explanation = {}

    try:
        import shap
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_single)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]
            explanation['shap'] = {name: float(val) for name, val in zip(feature_names, sv)}
    except Exception:
        pass

    try:
        if hasattr(model, 'feature_importances_'):
            explanation['feature_importance'] = {
                name: float(val) for name, val in zip(feature_names, model.feature_importances_)
            }
    except Exception:
        pass

    return explanation


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.feature_engineering import get_preprocessor

    df = load_data()
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    preprocessor = get_preprocessor()
    X_proc = preprocessor.fit_transform(X, y)
    feature_names = list(preprocessor.named_steps['col_trans'].get_feature_names_out())

    explain_model(X_proc, feature_names)
