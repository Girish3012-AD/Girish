"""
explain.py
==========
Model Explainability for Customer Churn.

Includes:
- Global feature importance
- Local explainability (SHAP or permutation importance)
- Risk scoring

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib

# Import config
try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR

PLOTS_DIR = OUTPUT_DIR / 'plots'
MODELS_DIR = OUTPUT_DIR / 'models'

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available, will use permutation importance")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_best_model(filepath: Path = None):
    """Load the best trained model."""
    if filepath is None:
        filepath = MODELS_DIR / 'best_model.pkl'
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    return joblib.load(filepath)


# =============================================================================
# RISK SCORING
# =============================================================================

def compute_risk_scores(model, X: pd.DataFrame, 
                        customer_ids: pd.Series = None,
                        threshold: float = 0.5) -> pd.DataFrame:
    """
    Compute churn risk scores for all customers.
    
    Args:
        model: Trained model pipeline
        X: Feature DataFrame
        customer_ids: Series of customer IDs
        threshold: Classification threshold
    
    Returns:
        DataFrame with customer_id, churn_probability, churn_prediction
    """
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'customer_id': customer_ids if customer_ids is not None else range(len(X)),
        'churn_probability': y_proba,
        'churn_prediction': y_pred
    })
    
    # Add risk category
    result['risk_category'] = pd.cut(
        result['churn_probability'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return result.sort_values('churn_probability', ascending=False)


def save_risk_scores(scores_df: pd.DataFrame, 
                     filepath: Path = None) -> Path:
    """Save risk scores to CSV."""
    if filepath is None:
        filepath = OUTPUT_DIR / 'customer_scores.csv'
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(filepath, index=False)
    print(f"✓ Saved customer scores: {filepath}")
    
    return filepath


# =============================================================================
# GLOBAL FEATURE IMPORTANCE
# =============================================================================

def get_global_feature_importance(model, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Extract global feature importance from model.
    
    Args:
        model: Trained model pipeline
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance
    """
    # Get the classifier from pipeline
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('classifier')
        preprocessor = model.named_steps.get('preprocessor')
    else:
        classifier = model
        preprocessor = None
    
    # Get feature names after preprocessing
    if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
        try:
            transformed_names = list(preprocessor.get_feature_names_out())
        except:
            transformed_names = feature_names if feature_names else [f'feature_{i}' for i in range(100)]
    else:
        transformed_names = feature_names if feature_names else [f'feature_{i}' for i in range(100)]
    
    # Extract importance
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_).flatten()
    else:
        return pd.DataFrame({'feature': ['Unknown'], 'importance': [0]})
    
    # Create DataFrame
    n_features = len(importances)
    if len(transformed_names) < n_features:
        transformed_names = [f'feature_{i}' for i in range(n_features)]
    
    importance_df = pd.DataFrame({
        'feature': transformed_names[:n_features],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_global_importance(importance_df: pd.DataFrame, 
                           top_n: int = 15,
                           plots_dir: Path = None) -> Path:
    """
    Plot global feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'].values,
                   color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Global Feature Importances')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = plots_dir / 'global_feature_importance.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: global_feature_importance.png")
    return filepath


# =============================================================================
# LOCAL EXPLAINABILITY
# =============================================================================

def explain_with_shap(model, X: pd.DataFrame, 
                      n_samples: int = 100) -> Tuple[Any, np.ndarray]:
    """
    Compute SHAP values for local explainability.
    
    Args:
        model: Trained model pipeline
        X: Feature DataFrame
        n_samples: Number of background samples
    
    Returns:
        (shap_explainer, shap_values)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed")
    
    # Sample background data
    background = X.sample(n=min(n_samples, len(X)), random_state=42)
    
    # Create explainer
    if hasattr(model, 'named_steps'):
        # For pipeline, we need to transform first
        preprocessor = model.named_steps.get('preprocessor')
        classifier = model.named_steps.get('classifier')
        
        if preprocessor is not None:
            background_transformed = preprocessor.transform(background)
            X_transformed = preprocessor.transform(X)
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
        else:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    
    return explainer, shap_values


def compute_permutation_importance(model, X: pd.DataFrame, y: pd.Series,
                                   n_repeats: int = 10) -> pd.DataFrame:
    """
    Compute permutation importance as alternative to SHAP.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        n_repeats: Number of permutation repeats
    
    Returns:
        DataFrame with permutation importance
    """
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(model, X, y, n_repeats=n_repeats, 
                                     random_state=42, n_jobs=-1)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df


def explain_high_risk_customers(model, X: pd.DataFrame, 
                                 risk_scores: pd.DataFrame,
                                 top_n: int = 5,
                                 feature_importance: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate explanations for top high-risk customers.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        risk_scores: DataFrame with risk scores
        top_n: Number of customers to explain
        feature_importance: Global feature importance
    
    Returns:
        DataFrame with customer explanations
    """
    # Get top high-risk customers
    top_customers = risk_scores.head(top_n)
    
    explanations = []
    
    for idx, row in top_customers.iterrows():
        customer_id = row['customer_id']
        prob = row['churn_probability']
        
        # Get customer features
        customer_idx = risk_scores.index.get_loc(idx) if idx in risk_scores.index else 0
        
        # Create simple rule-based explanation
        explanation = {
            'customer_id': customer_id,
            'churn_probability': prob,
            'risk_category': row['risk_category'],
            'top_risk_factors': []
        }
        
        # Find notable feature values
        if feature_importance is not None and len(feature_importance) > 0:
            top_features = feature_importance.head(5)['feature'].tolist()
            explanation['key_features'] = ', '.join(top_features[:3])
        
        explanations.append(explanation)
    
    return pd.DataFrame(explanations)


def plot_high_risk_explanation(customer_data: Dict, 
                               feature_values: Dict,
                               plots_dir: Path = None) -> Path:
    """
    Plot explanation for a high-risk customer.
    
    Args:
        customer_data: Customer information
        feature_values: Dictionary of feature values
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create bar plot of feature values
    features = list(feature_values.keys())[:10]
    values = [feature_values[f] for f in features]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if v > np.mean(values) else '#3498db' for v in values]
    
    bars = ax.barh(features, values, color=colors, edgecolor='black')
    
    ax.set_xlabel('Value')
    ax.set_title(f"Customer {customer_data.get('customer_id', 'Unknown')} - "
                f"Risk: {customer_data.get('churn_probability', 0)*100:.1f}%")
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    customer_id = customer_data.get('customer_id', 'unknown')
    filepath = plots_dir / f'explanation_{customer_id}.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Explainability module loaded successfully.")
    print(f"SHAP available: {SHAP_AVAILABLE}")
