"""
build_bi_dataset.py
===================
Build BI-ready dataset by merging all analytics outputs.

Combines:
- cleaned_dataset.csv
- customer_scores.csv
- retention_actions.csv

Output: bi/tableau_ready_dataset.csv

Author: Data Analyst + BI Engineer
"""

import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
BI_DIR = PROJECT_ROOT / 'bi'

def build_bi_dataset():
    """Build comprehensive BI dataset."""
    
    print("=" * 60)
    print("BUILDING BI DATASET")
    print("=" * 60)
    
    # Load data
    print("\nLoading data files...")
    cleaned_df = pd.read_csv(OUTPUTS_DIR / 'cleaned_dataset.csv')
    scores_df = pd.read_csv(OUTPUTS_DIR / 'customer_scores.csv')
    actions_df = pd.read_csv(OUTPUTS_DIR / 'retention_actions.csv')
    
    print(f"  Cleaned data: {len(cleaned_df):,} rows")
    print(f"  Scores: {len(scores_df):,} rows")
    print(f"  Actions: {len(actions_df):,} rows")
    
    # Merge all datasets
    print("\nMerging datasets...")
    bi_df = cleaned_df.merge(scores_df, on='customer_id', how='left')
    bi_df = bi_df.merge(actions_df, on='customer_id', how='left')
    
    # Select and rename columns for BI
    bi_columns = [
        'customer_id',
        'churn_probability',
        'churn_prediction',
        'risk_category',
        'segment_id',
        'recommended_action',
        'plan_type',
        'monthly_price',
        'age',
        'gender',
        'location',
        'device_type',
        'acquisition_channel',
        'auto_renew',
        'total_sessions_30d',
        'avg_session_minutes_30d',
        'total_crashes_30d',
        'failed_payments_30d',
        'total_amount_success_30d',
        'support_tickets_30d',
        'avg_resolution_time_30d',
        'churn'
    ]
    
    # Filter to available columns
    available_cols = [c for c in bi_columns if c in bi_df.columns]
    bi_df = bi_df[available_cols]
    
    # Save
    BI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BI_DIR / 'tableau_ready_dataset.csv'
    bi_df.to_csv(output_path, index=False)
    
    print(f"\n✓ BI dataset created: {output_path}")
    print(f"  Shape: {bi_df.shape[0]:,} rows × {bi_df.shape[1]} columns")
    print(f"  Columns: {list(bi_df.columns)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total customers: {len(bi_df):,}")
    print(f"Avg churn probability: {bi_df['churn_probability'].mean()*100:.1f}%")
    print(f"High risk (>= 70%): {(bi_df['churn_probability'] >= 0.7).sum():,}")
    
    return bi_df

if __name__ == "__main__":
    build_bi_dataset()
