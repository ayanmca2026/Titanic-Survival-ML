"""Model evaluation: metrics computation and visualization generation."""

import os
import json
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — MUST be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from sklearn.model_selection import learning_curve

from src.utils import IMAGES_DIR, REPORTS_DIR, MODELS_DIR, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name="BestModel"):
    """Evaluate model and generate all evaluation plots."""
    y_pred = model.predict(X_test)
    
    # Get probabilities safely
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    # Compute metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    # Save metrics
    with open(os.path.join(MODELS_DIR, 'best_model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics: {metrics}")

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived'])
    logger.info(f"\nClassification Report:\n{report}")

    # --- Confusion Matrix ---
    try:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Did Not Survive', 'Survived'],
                    yticklabels=['Did Not Survive', 'Survived'])
        ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f'confusion_matrix_{model_name}.png'), dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved.")
    except Exception as e:
        logger.warning(f"Confusion matrix plot failed: {e}")

    # --- ROC Curve ---
    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='#00D4FF', lw=2,
                    label=f'AUC = {metrics.get("roc_auc", 0):.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.fill_between(fpr, tpr, alpha=0.1, color='#00D4FF')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, f'roc_curve_{model_name}.png'), dpi=150)
            plt.close(fig)
            logger.info("ROC curve saved.")
        except Exception as e:
            logger.warning(f"ROC curve plot failed: {e}")

    # --- PR Curve ---
    if y_proba is not None:
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall_vals, precision_vals, color='#00FF88', lw=2)
            ax.fill_between(recall_vals, precision_vals, alpha=0.1, color='#00FF88')
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(IMAGES_DIR, f'pr_curve_{model_name}.png'), dpi=150)
            plt.close(fig)
            logger.info("PR curve saved.")
        except Exception as e:
            logger.warning(f"PR curve plot failed: {e}")

    # --- Learning Curve ---
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_test, y_test, cv=3, scoring='accuracy',
            train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#00D4FF', label='Training')
        ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='#FF4B4B', label='Validation')
        ax.fill_between(train_sizes,
                        np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                        np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                        alpha=0.1, color='#00D4FF')
        ax.fill_between(train_sizes,
                        np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                        np.mean(test_scores, axis=1) + np.std(test_scores, axis=1),
                        alpha=0.1, color='#FF4B4B')
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, f'learning_curve_{model_name}.png'), dpi=150)
        plt.close(fig)
        logger.info("Learning curve saved.")
    except Exception as e:
        logger.warning(f"Learning curve plot failed: {e}")

    return metrics
