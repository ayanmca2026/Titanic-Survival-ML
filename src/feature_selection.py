"""Feature Selection module: RFE, SelectKBest, Mutual Information, Variance Threshold, Tree-based."""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_selection import (
    RFE, SelectKBest, mutual_info_classif,
    VarianceThreshold, chi2, f_classif
)
from sklearn.ensemble import RandomForestClassifier

from src.utils import IMAGES_DIR, MODELS_DIR, REPORTS_DIR, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def run_feature_selection(X, y, feature_names):
    """Run all feature selection methods and save results."""
    X_df = pd.DataFrame(X, columns=feature_names)
    results = {}

    # 1. Variance Threshold
    logger.info("Running Variance Threshold...")
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_df)
    features_vt = list(X_df.columns[vt.get_support()])
    results['variance_threshold'] = features_vt
    logger.info(f"  Variance Threshold: {len(features_vt)}/{len(feature_names)} features kept")

    # 2. SelectKBest with f_classif
    logger.info("Running SelectKBest (f_classif)...")
    k = min(15, len(feature_names))
    skb_f = SelectKBest(score_func=f_classif, k=k)
    skb_f.fit(X_df, y)
    features_fclass = list(X_df.columns[skb_f.get_support()])
    scores_fclass = {name: float(score) for name, score in zip(feature_names, skb_f.scores_) if not np.isnan(score)}
    results['selectkbest_fclassif'] = features_fclass

    # 3. SelectKBest with Mutual Information
    logger.info("Running SelectKBest (mutual_info)...")
    skb_mi = SelectKBest(score_func=mutual_info_classif, k=k)
    skb_mi.fit(X_df, y)
    features_mi = list(X_df.columns[skb_mi.get_support()])
    scores_mi = {name: float(score) for name, score in zip(feature_names, skb_mi.scores_)}
    results['mutual_information'] = features_mi

    # 4. RFE with Random Forest
    logger.info("Running Recursive Feature Elimination...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=k, step=1)
    rfe.fit(X_df, y)
    features_rfe = list(X_df.columns[rfe.get_support()])
    rfe_ranking = {name: int(rank) for name, rank in zip(feature_names, rfe.ranking_)}
    results['rfe'] = features_rfe

    # 5. Tree-based Feature Importance
    logger.info("Running Tree-based Feature Importance...")
    rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_full.fit(X_df, y)
    tree_importance = {name: float(imp) for name, imp in zip(feature_names, rf_full.feature_importances_)}
    sorted_tree = sorted(tree_importance.items(), key=lambda x: x[1], reverse=True)
    results['tree_importance'] = [name for name, _ in sorted_tree[:k]]

    # Consensus: features selected by 3+ methods
    all_selected = features_vt + features_fclass + features_mi + features_rfe + [name for name, _ in sorted_tree[:k]]
    from collections import Counter
    counts = Counter(all_selected)
    consensus = [feat for feat, count in counts.items() if count >= 3]
    results['consensus'] = consensus
    logger.info(f"Consensus features ({len(consensus)}): {consensus}")

    # Save results
    with open(os.path.join(MODELS_DIR, 'feature_selection_results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=str)

    # Generate comparison plot
    _plot_feature_selection(scores_mi, scores_fclass, tree_importance, feature_names)

    return results


def _plot_feature_selection(mi_scores, f_scores, tree_imp, feature_names):
    """Generate feature selection comparison plot."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Mutual Information
        sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:12]
        names, vals = zip(*sorted_mi) if sorted_mi else ([], [])
        axes[0].barh(range(len(names)), vals, color='#00D4FF')
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=8)
        axes[0].set_title('Mutual Information', fontweight='bold')
        axes[0].invert_yaxis()

        # F-classif
        sorted_f = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)[:12]
        names, vals = zip(*sorted_f) if sorted_f else ([], [])
        axes[1].barh(range(len(names)), vals, color='#00FF88')
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=8)
        axes[1].set_title('F-Classif Scores', fontweight='bold')
        axes[1].invert_yaxis()

        # Tree importance
        sorted_tree = sorted(tree_imp.items(), key=lambda x: x[1], reverse=True)[:12]
        names, vals = zip(*sorted_tree) if sorted_tree else ([], [])
        axes[2].barh(range(len(names)), vals, color='#FFB800')
        axes[2].set_yticks(range(len(names)))
        axes[2].set_yticklabels(names, fontsize=8)
        axes[2].set_title('Tree-Based Importance', fontweight='bold')
        axes[2].invert_yaxis()

        plt.suptitle('Feature Selection Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'feature_selection_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Feature selection comparison plot saved.")

    except Exception as e:
        logger.warning(f"Feature selection plot failed: {e}")


# Keep backward compatibility
def select_features(X, y, feature_names):
    """Backward-compatible wrapper."""
    results = run_feature_selection(X, y, feature_names)
    return results.get('consensus', feature_names)
