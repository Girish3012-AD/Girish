"""
features.py
===========
Feature Engineering for Customer Churn Model.

Includes:
- Interaction feature creation
- Log transforms for skewed features
- sklearn Pipeline with ColumnTransformer

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Import project config
try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR


# =============================================================================
# FEATURE CONFIGURATIONS
# =============================================================================

# Numeric features for scaling
NUMERIC_FEATURES = [
    'age', 'monthly_price', 'auto_renew',
    'total_sessions_30d', 'avg_session_minutes_30d', 'total_crashes_30d',
    'failed_payments_30d', 'total_amount_success_30d',
    'support_tickets_30d', 'avg_resolution_time_30d'
]

# Categorical features for encoding
CATEGORICAL_FEATURES = [
    'gender', 'location', 'device_type', 'acquisition_channel', 'plan_type'
]

# Features to log-transform (skewed features)
LOG_TRANSFORM_FEATURES = [
    'total_sessions_30d', 'avg_session_minutes_30d', 'total_crashes_30d',
    'failed_payments_30d', 'total_amount_success_30d',
    'support_tickets_30d', 'avg_resolution_time_30d'
]

# ID and target columns
ID_COL = 'customer_id'
TARGET_COL = 'churn'


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features.
    
    Features created:
    - sessions_per_crash: total_sessions_30d / (total_crashes_30d + 1)
    - payment_failure_rate: failed_payments_30d / (failed_payments_30d + 1 + total_amount_success_30d)
    - support_per_session: support_tickets_30d / (total_sessions_30d + 1)
    - avg_minutes_per_session: avg_session_minutes_30d / (total_sessions_30d + 1)
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with new features
    """
    df = df.copy()
    
    # Sessions per crash (higher = better stability)
    df['sessions_per_crash'] = df['total_sessions_30d'] / (df['total_crashes_30d'] + 1)
    
    # Payment failure rate
    df['payment_failure_rate'] = df['failed_payments_30d'] / (
        df['failed_payments_30d'] + 1 + df['total_amount_success_30d']
    )
    
    # Support tickets per session (engagement vs problems)
    df['support_per_session'] = df['support_tickets_30d'] / (df['total_sessions_30d'] + 1)
    
    # Average engagement per session
    df['avg_minutes_per_session'] = df['avg_session_minutes_30d'] / (df['total_sessions_30d'] + 1)
    
    print(f"Created interaction features: sessions_per_crash, payment_failure_rate, "
          f"support_per_session, avg_minutes_per_session")
    
    return df


def apply_log_transforms(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Apply log1p transform to skewed numeric columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to transform (default: LOG_TRANSFORM_FEATURES)
    
    Returns:
        DataFrame with log-transformed columns
    """
    df = df.copy()
    
    if columns is None:
        columns = LOG_TRANSFORM_FEATURES
    
    for col in columns:
        if col in df.columns:
            new_col = f'{col}_log'
            df[new_col] = np.log1p(df[col].clip(lower=0))
    
    print(f"Applied log1p transform to {len(columns)} columns")
    
    return df


def get_feature_names(include_interactions: bool = True,
                      include_log: bool = True) -> Tuple[List[str], List[str]]:
    """
    Get all feature names for the model.
    
    Args:
        include_interactions: Include interaction features
        include_log: Include log-transformed features
    
    Returns:
        (numeric_features, categorical_features)
    """
    numeric = NUMERIC_FEATURES.copy()
    
    if include_interactions:
        numeric.extend([
            'sessions_per_crash', 'payment_failure_rate',
            'support_per_session', 'avg_minutes_per_session'
        ])
    
    if include_log:
        numeric.extend([f'{col}_log' for col in LOG_TRANSFORM_FEATURES])
    
    return numeric, CATEGORICAL_FEATURES.copy()


# =============================================================================
# SKLEARN PIPELINE
# =============================================================================

def build_preprocessing_pipeline(numeric_features: List[str] = None,
                                  categorical_features: List[str] = None) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline with ColumnTransformer.
    
    Components:
    - Numeric: SimpleImputer (median) + StandardScaler
    - Categorical: SimpleImputer (most_frequent) + OneHotEncoder
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
    
    Returns:
        ColumnTransformer preprocessor
    """
    if numeric_features is None:
        numeric_features, categorical_features = get_feature_names(
            include_interactions=True, include_log=True
        )
    
    if categorical_features is None:
        _, categorical_features = get_feature_names()
    
    # Numeric preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    print(f"Built preprocessing pipeline:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    
    return preprocessor


def prepare_features(df: pd.DataFrame,
                     create_interactions: bool = True,
                     apply_log: bool = True) -> pd.DataFrame:
    """
    Prepare features for modeling.
    
    Args:
        df: Input DataFrame
        create_interactions: Whether to create interaction features
        apply_log: Whether to apply log transforms
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    if create_interactions:
        df = create_interaction_features(df)
    
    if apply_log:
        df = apply_log_transforms(df)
    
    return df


def get_X_y(df: pd.DataFrame,
            numeric_features: List[str] = None,
            categorical_features: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        numeric_features: Numeric feature columns
        categorical_features: Categorical feature columns
    
    Returns:
        X (features DataFrame), y (target Series)
    """
    if numeric_features is None:
        numeric_features, categorical_features = get_feature_names(
            include_interactions=True, include_log=True
        )
    
    if categorical_features is None:
        _, categorical_features = get_feature_names()
    
    all_features = numeric_features + categorical_features
    
    # Filter to columns that exist
    available_features = [col for col in all_features if col in df.columns]
    
    X = df[available_features]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    
    return X, y


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Feature Engineering module loaded successfully.")
    
    # Print configuration
    print("\nNumeric Features:")
    for f in NUMERIC_FEATURES:
        print(f"  - {f}")
    
    print("\nCategorical Features:")
    for f in CATEGORICAL_FEATURES:
        print(f"  - {f}")
