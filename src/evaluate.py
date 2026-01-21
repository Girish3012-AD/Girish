"""
evaluate.py
===========
Model Evaluation and Visualization for Customer Churn.

Includes:
- ROC and PR curve plotting
- Confusion matrix visualization
- Metric calculation and reporting

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score
)

# Import config
try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR

PLOTS_DIR = OUTPUT_DIR / 'plots'


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   model_name: str = 'Model',
                   plots_dir: Path = None) -> Path:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#e74c3c', lw=2,
            label=f'{model_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color='#e74c3c')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    filepath = plots_dir / 'roc_curve.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: roc_curve.png")
    return filepath


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                                 model_name: str = 'Model',
                                 plots_dir: Path = None) -> Path:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='#3498db', lw=2,
            label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1, 
               label=f'Baseline = {baseline:.3f}')
    
    ax.fill_between(recall, precision, alpha=0.2, color='#3498db')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    filepath = plots_dir / 'precision_recall_curve.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: precision_recall_curve.png")
    return filepath


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          threshold: float = 0.5,
                          plots_dir: Path = None) -> Path:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        threshold: Classification threshold used
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, cmap='Blues')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Count')
    
    # Labels
    labels = ['Not Churned', 'Churned']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = 'white' if value > cm.max() / 2 else 'black'
            ax.text(j, i, f'{value:,}', ha='center', va='center',
                   color=color, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    
    filepath = plots_dir / f'confusion_matrix_t{int(threshold*100)}.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: confusion_matrix_t{int(threshold*100)}.png")
    return filepath


def plot_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 15,
                            plots_dir: Path = None) -> Path:
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()  # Top feature at top
    
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = plots_dir / 'feature_importance.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: feature_importance.png")
    return filepath


def plot_threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                            plots_dir: Path = None) -> Path:
    """
    Plot metrics vs threshold analysis.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, 'b-', label='Precision', lw=2)
    ax.plot(thresholds, recalls, 'g-', label='Recall', lw=2)
    ax.plot(thresholds, f1s, 'r-', label='F1', lw=2)
    
    # Mark best F1 threshold
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.5,
               label=f'Best F1 @ {best_threshold:.2f}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Classification Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filepath = plots_dir / 'threshold_analysis.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: threshold_analysis.png")
    return filepath


# =============================================================================
# REPORTING FUNCTIONS
# =============================================================================

def print_model_summary(model_name: str, metrics: Dict[str, float],
                        feature_importance: pd.DataFrame = None,
                        top_n_features: int = 10) -> None:
    """
    Print comprehensive model summary.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of metrics
        feature_importance: Optional DataFrame with feature importances
        top_n_features: Number of top features to show
    """
    print("\n" + "=" * 60)
    print("BEST MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Model: {model_name}")
    print(f"\n📈 Performance Metrics:")
    print(f"   • ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
    print(f"   • PR-AUC: {metrics.get('pr_auc', 0):.4f}")
    print(f"   • Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   • Precision: {metrics.get('precision', 0):.4f}")
    print(f"   • Recall: {metrics.get('recall', 0):.4f}")
    print(f"   • F1: {metrics.get('f1', 0):.4f}")
    print(f"   • Best Threshold: {metrics.get('best_threshold', 0.5):.2f}")
    
    if feature_importance is not None and len(feature_importance) > 0:
        print(f"\n🔝 Top {top_n_features} Features:")
        for i, row in feature_importance.head(top_n_features).iterrows():
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
