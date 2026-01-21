"""
data_quality_checks.py
======================
Data quality and leakage detection for Customer Churn project.

Performs comprehensive checks including:
- Missing values analysis
- Duplicate detection
- Outlier detection (IQR method)
- Impossible value detection
- Data leakage checks

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import project config
try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR


# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

def check_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """Return DataFrame shape (rows, cols)."""
    return df.shape


def check_duplicates(df: pd.DataFrame, id_col: str = 'customer_id') -> int:
    """Count duplicate values in ID column."""
    if id_col not in df.columns:
        return 0
    return df[id_col].duplicated().sum()


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing values per column.
    
    Returns:
        DataFrame with missing counts and percentages
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    result = pd.DataFrame({
        'column': missing.index,
        'missing_count': missing.values,
        'missing_pct': missing_pct.values
    })
    
    return result[result['missing_count'] > 0].sort_values('missing_count', ascending=False)


def check_unique_values(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict]:
    """
    Get unique value info for categorical columns.
    
    Returns:
        Dictionary with unique counts and values
    """
    result = {}
    for col in cat_cols:
        if col in df.columns:
            unique_vals = df[col].unique()
            result[col] = {
                'n_unique': len(unique_vals),
                'values': unique_vals.tolist()[:10]  # Limit to 10 values
            }
    return result


def compute_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic statistics for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.describe().T


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str] = None,
                        multiplier: float = 1.5) -> Dict[str, Dict]:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        numeric_cols: Columns to check (default: all numeric)
        multiplier: IQR multiplier (1.5 = standard, 3 = extreme)
    
    Returns:
        Dictionary with outlier info per column
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            outliers[col] = {
                'count': int(n_outliers),
                'pct': round(n_outliers / len(data) * 100, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'min': round(data.min(), 2),
                'max': round(data.max(), 2)
            }
    
    return outliers


def detect_impossible_values(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Detect impossible/invalid values based on business rules.
    
    Rules:
    - age: must be between 10 and 100
    - monthly_price: must be >= 0
    - sessions: must be >= 0
    - crashes: must be >= 0
    - failed_payments: must be >= 0
    """
    issues = {}
    
    # Age validation
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 10) | (df['age'] > 100)).sum()
        if invalid_age > 0:
            issues['age'] = {
                'count': int(invalid_age),
                'rule': 'age must be between 10 and 100',
                'invalid_values': df.loc[(df['age'] < 10) | (df['age'] > 100), 'age'].unique().tolist()[:5]
            }
    
    # Monthly price validation
    if 'monthly_price' in df.columns:
        invalid_price = (df['monthly_price'] < 0).sum()
        if invalid_price > 0:
            issues['monthly_price'] = {
                'count': int(invalid_price),
                'rule': 'monthly_price must be >= 0',
                'min_value': float(df['monthly_price'].min())
            }
    
    # Negative values in count columns
    count_cols = ['total_sessions_30d', 'total_crashes_30d', 'failed_payments_30d',
                  'support_tickets_30d']
    for col in count_cols:
        if col in df.columns:
            invalid = (df[col] < 0).sum()
            if invalid > 0:
                issues[col] = {
                    'count': int(invalid),
                    'rule': f'{col} must be >= 0'
                }
    
    return issues


# =============================================================================
# LEAKAGE DETECTION
# =============================================================================

def check_leakage(df: pd.DataFrame, target_col: str = 'churn') -> List[Dict]:
    """
    Check for potential data leakage issues.
    
    Checks:
    1. is_active/end_date columns that directly encode target
    2. Features with perfect correlation to target
    3. Columns that shouldn't exist in training data
    
    Returns:
        List of leakage warnings
    """
    warnings = []
    
    # Check for is_active column (direct leakage)
    if 'is_active' in df.columns:
        correlation = df['is_active'].corr(df[target_col])
        if abs(correlation) > 0.9:
            warnings.append({
                'type': 'CRITICAL',
                'feature': 'is_active',
                'issue': f'Direct leakage detected! Correlation with churn: {correlation:.3f}',
                'action': 'REMOVE this feature before training'
            })
    
    # Check for end_date column (direct leakage)
    if 'end_date' in df.columns:
        warnings.append({
            'type': 'CRITICAL',
            'feature': 'end_date',
            'issue': 'end_date directly reveals churn status',
            'action': 'REMOVE this feature before training'
        })
    
    # Check for churn_date column (direct leakage)
    if 'churn_date' in df.columns:
        warnings.append({
            'type': 'CRITICAL',
            'feature': 'churn_date',
            'issue': 'churn_date directly reveals churn status',
            'action': 'REMOVE this feature before training'
        })
    
    # Check for high correlation features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    for col in numeric_cols:
        corr = df[col].corr(df[target_col])
        if abs(corr) > 0.8:
            warnings.append({
                'type': 'WARNING',
                'feature': col,
                'issue': f'High correlation with target: {corr:.3f}',
                'action': 'Investigate if this feature uses future information'
            })
    
    # Check feature distributions by churn status
    for col in numeric_cols:
        churned = df[df[target_col] == 1][col].mean()
        not_churned = df[df[target_col] == 0][col].mean()
        
        # Check for suspiciously large difference
        if not_churned != 0:
            ratio = churned / not_churned
            if ratio > 5 or ratio < 0.2:
                warnings.append({
                    'type': 'INFO',
                    'feature': col,
                    'issue': f'Large mean difference: churned={churned:.2f}, not_churned={not_churned:.2f}',
                    'action': 'Verify this is a legitimate predictive signal'
                })
    
    return warnings


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_eda_report(df: pd.DataFrame, output_path: Path = None) -> str:
    """
    Generate comprehensive EDA report.
    
    Args:
        df: Input DataFrame
        output_path: Path to save report
    
    Returns:
        Report as string
    """
    if output_path is None:
        output_path = OUTPUT_DIR / 'eda_report.txt'
    
    lines = []
    lines.append("=" * 70)
    lines.append("CUSTOMER CHURN EDA REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    # Shape
    rows, cols = check_shape(df)
    lines.append(f"\n📊 DATASET SHAPE")
    lines.append(f"   Rows: {rows:,}")
    lines.append(f"   Columns: {cols}")
    
    # Duplicates
    dupes = check_duplicates(df)
    lines.append(f"\n🔍 DUPLICATE CHECK")
    lines.append(f"   Duplicate customer_ids: {dupes}")
    
    # Missing values
    missing = check_missing_values(df)
    lines.append(f"\n❓ MISSING VALUES")
    if len(missing) == 0:
        lines.append("   No missing values found!")
    else:
        for _, row in missing.iterrows():
            lines.append(f"   {row['column']}: {row['missing_count']} ({row['missing_pct']}%)")
    
    # Categorical unique values
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    lines.append(f"\n📋 CATEGORICAL COLUMNS")
    unique_info = check_unique_values(df, cat_cols)
    for col, info in unique_info.items():
        lines.append(f"   {col}: {info['n_unique']} unique values")
        lines.append(f"      Values: {info['values']}")
    
    # Numeric stats
    lines.append(f"\n📈 NUMERIC STATISTICS")
    stats = compute_numeric_stats(df)
    lines.append(stats.to_string())
    
    # Outliers
    lines.append(f"\n⚠️ OUTLIERS (IQR Method)")
    outliers = detect_outliers_iqr(df)
    if len(outliers) == 0:
        lines.append("   No significant outliers detected!")
    else:
        for col, info in outliers.items():
            lines.append(f"   {col}: {info['count']} outliers ({info['pct']}%)")
            lines.append(f"      Bounds: [{info['lower_bound']}, {info['upper_bound']}]")
            lines.append(f"      Actual: [{info['min']}, {info['max']}]")
    
    # Impossible values
    lines.append(f"\n🚫 IMPOSSIBLE VALUES")
    impossible = detect_impossible_values(df)
    if len(impossible) == 0:
        lines.append("   No impossible values detected!")
    else:
        for col, info in impossible.items():
            lines.append(f"   {col}: {info['count']} invalid values")
            lines.append(f"      Rule: {info['rule']}")
    
    # Leakage checks
    lines.append(f"\n🔒 LEAKAGE DETECTION")
    leakage_warnings = check_leakage(df)
    if len(leakage_warnings) == 0:
        lines.append("   No data leakage detected!")
    else:
        for warning in leakage_warnings:
            lines.append(f"   [{warning['type']}] {warning['feature']}")
            lines.append(f"      Issue: {warning['issue']}")
            lines.append(f"      Action: {warning['action']}")
    
    # Churn distribution
    if 'churn' in df.columns:
        churn_rate = df['churn'].mean() * 100
        lines.append(f"\n🎯 TARGET DISTRIBUTION")
        lines.append(f"   Churn rate: {churn_rate:.2f}%")
        lines.append(f"   Churned: {df['churn'].sum():,}")
        lines.append(f"   Not churned: {(df['churn'] == 0).sum():,}")
    
    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ EDA report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    # Test with sample data
    print("Data Quality Checks module loaded successfully.")
