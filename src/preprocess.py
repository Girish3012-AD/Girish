"""
preprocess.py
=============
Data preprocessing and cleaning for Customer Churn project.

Includes:
- Missing value imputation
- Outlier capping (percentile clipping)
- Column standardization
- Train/test split with stratification

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

# Import project config
try:
    from config import OUTPUT_DIR, PROJECT_ROOT
except ImportError:
    from src.config import OUTPUT_DIR, PROJECT_ROOT


# =============================================================================
# CONFIGURATION
# =============================================================================

# Columns that should be numeric
NUMERIC_COLS = [
    'age', 'monthly_price', 'auto_renew',
    'total_sessions_30d', 'avg_session_minutes_30d', 'total_crashes_30d',
    'failed_payments_30d', 'total_amount_success_30d',
    'support_tickets_30d', 'avg_resolution_time_30d'
]

# Columns that should be categorical
CATEGORICAL_COLS = [
    'gender', 'location', 'device_type', 'acquisition_channel', 'plan_type'
]

# ID column (not a feature)
ID_COL = 'customer_id'

# Target column
TARGET_COL = 'churn'


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lowercase with underscores.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = [col.lower().strip().replace(' ', '_').replace('-', '_') 
                  for col in df.columns]
    return df


def fill_missing_values(df: pd.DataFrame,
                        numeric_strategy: str = 'median',
                        categorical_fill: str = 'Unknown') -> pd.DataFrame:
    """
    Fill missing values.
    
    Args:
        df: Input DataFrame
        numeric_strategy: 'median' or 'mean' for numeric columns
        categorical_fill: Value to fill for categorical columns
    
    Returns:
        DataFrame with filled missing values
    """
    df = df.copy()
    
    # Fill numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            if numeric_strategy == 'median':
                fill_value = df[col].median()
            else:
                fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)
            print(f"  Filled {col} with {numeric_strategy}: {fill_value:.2f}")
    
    # Fill categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(categorical_fill)
            print(f"  Filled {col} with: '{categorical_fill}'")
    
    return df


def cap_outliers(df: pd.DataFrame, 
                 lower_percentile: float = 0.01,
                 upper_percentile: float = 0.99,
                 columns: List[str] = None) -> pd.DataFrame:
    """
    Cap outliers using percentile clipping.
    
    Args:
        df: Input DataFrame
        lower_percentile: Lower percentile for clipping (default 1%)
        upper_percentile: Upper percentile for clipping (default 99%)
        columns: Columns to cap (default: all numeric)
    
    Returns:
        DataFrame with capped outliers
    """
    df = df.copy()
    
    if columns is None:
        columns = [col for col in NUMERIC_COLS if col in df.columns]
    
    for col in columns:
        if col not in df.columns:
            continue
        
        lower_val = df[col].quantile(lower_percentile)
        upper_val = df[col].quantile(upper_percentile)
        
        original_min = df[col].min()
        original_max = df[col].max()
        
        df[col] = df[col].clip(lower=lower_val, upper=upper_val)
        
        if original_min < lower_val or original_max > upper_val:
            print(f"  Capped {col}: [{lower_val:.2f}, {upper_val:.2f}]")
    
    return df


def ensure_correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns have correct data types.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with correct dtypes
    """
    df = df.copy()
    
    # Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Target column
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    
    # ID column
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)
    
    return df


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    
    Args:
        df: Input DataFrame
        verbose: Print progress
    
    Returns:
        Cleaned DataFrame
    """
    if verbose:
        print("=" * 50)
        print("CLEANING DATASET")
        print("=" * 50)
    
    # Step 1: Standardize column names
    if verbose:
        print("\n1. Standardizing column names...")
    df = standardize_column_names(df)
    
    # Step 2: Ensure correct dtypes
    if verbose:
        print("\n2. Ensuring correct data types...")
    df = ensure_correct_dtypes(df)
    
    # Step 3: Fill missing values
    if verbose:
        print("\n3. Filling missing values...")
    df = fill_missing_values(df)
    
    # Step 4: Cap outliers
    if verbose:
        print("\n4. Capping outliers (1% - 99% percentiles)...")
    df = cap_outliers(df)
    
    if verbose:
        print("\n✓ Cleaning complete!")
        print(f"  Final shape: {df.shape}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
    
    return df


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_raw_data(filepath: Path = None) -> pd.DataFrame:
    """
    Load the raw/final churn dataset.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame
    """
    if filepath is None:
        filepath = OUTPUT_DIR / 'final_churn_dataset.csv'
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


def load_clean_data(filepath: Path = None) -> pd.DataFrame:
    """
    Load the cleaned dataset.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame
    """
    if filepath is None:
        filepath = OUTPUT_DIR / 'cleaned_dataset.csv'
    
    if not filepath.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded cleaned dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


def save_clean_data(df: pd.DataFrame, filepath: Path = None) -> Path:
    """
    Save cleaned dataset.
    
    Args:
        df: DataFrame to save
        filepath: Output path
    
    Returns:
        Path where file was saved
    """
    if filepath is None:
        filepath = OUTPUT_DIR / 'cleaned_dataset.csv'
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved cleaned dataset: {filepath}")
    
    return filepath


# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

def split_data(df: pd.DataFrame = None,
               test_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame (loads cleaned data if None)
        test_size: Proportion for test set
        random_state: Random seed
        stratify: Whether to stratify by target
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if df is None:
        df = load_clean_data()
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [ID_COL, TARGET_COL]]
    
    X = df[feature_cols]
    y = df[TARGET_COL]
    
    # Stratify by target if specified
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Train churn rate: {y_train.mean()*100:.2f}%")
    print(f"  Test churn rate: {y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Get lists of numeric and categorical feature columns.
    
    Returns:
        (numeric_cols, categorical_cols)
    """
    return NUMERIC_COLS.copy(), CATEGORICAL_COLS.copy()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to clean dataset and save."""
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load raw data
    df = load_raw_data()
    
    # Clean dataset
    df_clean = clean_dataset(df)
    
    # Save cleaned dataset
    save_clean_data(df_clean)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cleaned dataset shape: {df_clean.shape}")
    print(f"Churn rate: {df_clean['churn'].mean()*100:.2f}%")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    
    # Test split
    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    print("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    main()
