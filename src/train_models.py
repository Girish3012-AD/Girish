"""
train_models.py
===============
Model Training for Customer Churn Prediction.

Trains and compares:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost (if available)

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will skip XGB model")

# Import project modules
try:
    from config import OUTPUT_DIR
    from features import build_preprocessing_pipeline, get_feature_names
except ImportError:
    from src.config import OUTPUT_DIR
    from src.features import build_preprocessing_pipeline, get_feature_names


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS_DIR = OUTPUT_DIR / 'models'


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.
    
    Returns:
        Dictionary mapping model names to model objects
    """
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42,
            class_weight='balanced', n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42,
            learning_rate=0.1
        ),
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=5, random_state=42,
            learning_rate=0.1, use_label_encoder=False,
            eval_metric='logloss', scale_pos_weight=3
        )
    
    return models


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                preprocessor=None) -> Pipeline:
    """
    Train a model with optional preprocessing.
    
    Args:
        model: sklearn-compatible model
        X_train: Training features
        y_train: Training labels
        preprocessor: Optional preprocessor pipeline
    
    Returns:
        Trained Pipeline (preprocessor + model)
    """
    if preprocessor is not None:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', model)
        ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series,
                         preprocessor=None, cv: int = 5) -> Dict[str, float]:
    """
    Cross-validate a model.
    
    Args:
        model: sklearn-compatible model
        X: Features
        y: Labels
        preprocessor: Optional preprocessor
        cv: Number of folds
    
    Returns:
        Dictionary of cross-validation scores
    """
    if preprocessor is not None:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    else:
        pipeline = model
    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Compute CV scores for different metrics
    scores = {}
    
    for scoring in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        cv_scores = cross_val_score(pipeline, X, y, cv=cv_splitter, scoring=scoring)
        scores[f'{scoring}_mean'] = cv_scores.mean()
        scores[f'{scoring}_std'] = cv_scores.std()
    
    return scores


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        pipeline: Trained pipeline
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    # Predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'threshold': threshold
    }
    
    return metrics


def find_best_threshold(pipeline: Pipeline, X_val: pd.DataFrame, 
                        y_val: pd.Series, metric: str = 'f1') -> Tuple[float, float]:
    """
    Find best threshold based on specified metric.
    
    Args:
        pipeline: Trained pipeline
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('f1', 'recall', 'precision')
    
    Returns:
        (best_threshold, best_score)
    """
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        else:
            score = f1_score(y_val, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def get_feature_importance(pipeline: Pipeline, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        pipeline: Trained pipeline
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importances
    """
    model = pipeline.named_steps['classifier']
    
    # Get transformed feature names from preprocessor
    preprocessor = pipeline.named_steps.get('preprocessor')
    if preprocessor is not None:
        try:
            # Get feature names after transformation
            if hasattr(preprocessor, 'get_feature_names_out'):
                transformed_names = preprocessor.get_feature_names_out()
            else:
                transformed_names = feature_names
        except:
            transformed_names = [f'feature_{i}' for i in range(100)]
    else:
        transformed_names = feature_names
    
    # Extract importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        return pd.DataFrame()
    
    # Match lengths
    n_features = len(importances)
    if len(transformed_names) != n_features:
        transformed_names = [f'feature_{i}' for i in range(n_features)]
    
    importance_df = pd.DataFrame({
        'feature': transformed_names[:n_features],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def train_and_compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_train: pd.Series, y_test: pd.Series,
                             preprocessor=None) -> Tuple[Dict, pd.DataFrame, Pipeline]:
    """
    Train and compare all models.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        preprocessor: Preprocessing pipeline
    
    Returns:
        (all_metrics, comparison_df, best_pipeline)
    """
    models = get_models()
    all_metrics = {}
    comparison_results = []
    best_pipeline = None
    best_roc_auc = 0.0
    
    print("=" * 60)
    print("MODEL TRAINING AND COMPARISON")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Clone preprocessor for each model
        if preprocessor is not None:
            from sklearn.base import clone
            model_preprocessor = clone(preprocessor)
        else:
            model_preprocessor = None
        
        # Train model
        pipeline = train_model(model, X_train, y_train, model_preprocessor)
        
        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)
        all_metrics[name] = metrics
        
        # Find best threshold
        best_threshold, best_f1 = find_best_threshold(pipeline, X_test, y_test)
        metrics['best_threshold'] = best_threshold
        metrics['best_f1'] = best_f1
        
        # Track results
        comparison_results.append({
            'model': name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
            'best_threshold': best_threshold
        })
        
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Best Threshold: {best_threshold:.2f}")
        
        # Update best model
        if metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_pipeline = pipeline
            best_model_name = name
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
    
    print(f"\n✓ Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    return all_metrics, comparison_df, best_pipeline


# =============================================================================
# SAVE/LOAD FUNCTIONS
# =============================================================================

def save_model(pipeline: Pipeline, filepath: Path = None) -> Path:
    """Save model to file."""
    if filepath is None:
        filepath = MODELS_DIR / 'best_model.pkl'
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"✓ Model saved: {filepath}")
    
    return filepath


def load_model(filepath: Path = None) -> Pipeline:
    """Load model from file."""
    if filepath is None:
        filepath = MODELS_DIR / 'best_model.pkl'
    
    return joblib.load(filepath)


def save_metrics(metrics: Dict, filepath: Path = None) -> Path:
    """Save metrics to JSON file."""
    if filepath is None:
        filepath = OUTPUT_DIR / 'metrics.json'
    
    # Convert numpy types to Python types
    metrics_clean = {}
    for model, model_metrics in metrics.items():
        metrics_clean[model] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in model_metrics.items()
        }
    
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    
    print(f"✓ Metrics saved: {filepath}")
    return filepath


def save_comparison(df: pd.DataFrame, filepath: Path = None) -> Path:
    """Save comparison DataFrame to CSV."""
    if filepath is None:
        filepath = OUTPUT_DIR / 'model_comparison.csv'
    
    df.to_csv(filepath, index=False)
    print(f"✓ Comparison saved: {filepath}")
    return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Model Training module loaded successfully.")
    print(f"\nAvailable models: {list(get_models().keys())}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
