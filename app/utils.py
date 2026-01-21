"""
utils.py
========
Utility functions for FastAPI churn prediction API.

Author: Senior ML Engineer
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get project paths
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'models'
MODEL_PATH = MODELS_DIR / 'best_model.pkl'

# Prediction threshold
CHURN_THRESHOLD = 0.5

# Feature columns (must match training)
FEATURE_COLUMNS = [
    'age', 'gender', 'location', 'device_type', 'acquisition_channel',
    'plan_type', 'monthly_price', 'auto_renew',
    'total_sessions_30d', 'avg_session_minutes_30d', 'total_crashes_30d',
    'failed_payments_30d', 'total_amount_success_30d',
    'support_tickets_30d', 'avg_resolution_time_30d'
]

# Numeric columns for feature engineering
NUMERIC_COLS = [
    'age', 'monthly_price', 'auto_renew',
    'total_sessions_30d', 'avg_session_minutes_30d', 'total_crashes_30d',
    'failed_payments_30d', 'total_amount_success_30d',
    'support_tickets_30d', 'avg_resolution_time_30d'
]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: Path = None):
    """
    Load the trained model pipeline.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Loaded model pipeline or None if failed
    """
    if model_path is None:
        model_path = MODEL_PATH
    
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features for prediction."""
    df = df.copy()
    
    # Sessions per crash
    df['sessions_per_crash'] = df['total_sessions_30d'] / (df['total_crashes_30d'] + 1)
    
    # Payment failure rate
    df['payment_failure_rate'] = df['failed_payments_30d'] / (
        df['failed_payments_30d'] + 1 + df['total_amount_success_30d']
    )
    
    # Support per session
    df['support_per_session'] = df['support_tickets_30d'] / (df['total_sessions_30d'] + 1)
    
    # Avg minutes per session
    df['avg_minutes_per_session'] = df['avg_session_minutes_30d'] / (df['total_sessions_30d'] + 1)
    
    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log transforms to skewed features."""
    df = df.copy()
    
    log_features = [
        'total_sessions_30d', 'avg_session_minutes_30d', 'total_crashes_30d',
        'failed_payments_30d', 'total_amount_success_30d',
        'support_tickets_30d', 'avg_resolution_time_30d'
    ]
    
    for col in log_features:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    
    return df


def prepare_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare features for model prediction.
    
    Args:
        data: Dictionary of customer data
    
    Returns:
        DataFrame ready for prediction
    """
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Ensure all required columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col in NUMERIC_COLS else 'Unknown'
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Apply log transforms
    df = apply_log_transforms(df)
    
    return df


# =============================================================================
# PREDICTION
# =============================================================================

def get_risk_category(probability: float) -> str:
    """Get risk category based on churn probability."""
    if probability >= 0.7:
        return "Very High"
    elif probability >= 0.5:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    else:
        return "Low"


def predict_single(model, data: Dict[str, Any], threshold: float = None) -> Dict[str, Any]:
    """
    Make prediction for a single customer.
    
    Args:
        model: Trained model pipeline
        data: Customer data dictionary
        threshold: Classification threshold
    
    Returns:
        Dictionary with prediction results
    """
    if threshold is None:
        threshold = CHURN_THRESHOLD
    
    # Prepare features
    df = prepare_features(data)
    
    # Get prediction
    proba = model.predict_proba(df)[0, 1]
    prediction = int(proba >= threshold)
    category = get_risk_category(proba)
    
    return {
        "churn_probability": round(float(proba), 4),
        "churn_prediction": prediction,
        "risk_category": category
    }


def predict_batch(model, data_list: list, threshold: float = None) -> Dict[str, Any]:
    """
    Make predictions for multiple customers.
    
    Args:
        model: Trained model pipeline
        data_list: List of customer data dictionaries
        threshold: Classification threshold
    
    Returns:
        Dictionary with batch prediction results
    """
    if threshold is None:
        threshold = CHURN_THRESHOLD
    
    predictions = []
    high_risk_count = 0
    
    for data in data_list:
        result = predict_single(model, data, threshold)
        predictions.append(result)
        if result["churn_probability"] >= 0.7:
            high_risk_count += 1
    
    return {
        "predictions": predictions,
        "total_customers": len(predictions),
        "high_risk_count": high_risk_count
    }
