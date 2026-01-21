"""
monitor.py
==========
Model monitoring utilities for tracking predictions and performance.

Features:
- Baseline statistics generation
- Prediction logging
- Performance metrics tracking

Author: MLOps Engineer
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MONITORING_DIR = PROJECT_ROOT / 'monitoring'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

def generate_baseline_stats(df: pd.DataFrame, output_path: Path = None):
    """
    Generate baseline statistics from training/validation data.
    
    Args:
        df: DataFrame with features and predictions
        output_path: Where to save baseline stats
    
    Returns:
        dict: Baseline statistics
    """
    if output_path is None:
        output_path = MONITORING_DIR / 'baseline_stats.json'
    
    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    baseline = {
        'created_at': datetime.now().isoformat(),
        'n_samples': len(df),
        'feature_stats': {},
        'target_distribution': {}
    }
    
    # Numeric feature stats
    for col in numeric_cols:
        if col == 'churn':
            continue
        baseline['feature_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75))
        }
    
    # Target distribution
    if 'churn' in df.columns:
        baseline['target_distribution'] = {
            'churn_rate': float(df['churn'].mean()),
            'churn_count': int(df['churn'].sum()),
            'total_count': int(len(df))
        }
    
    # Save
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"✓ Baseline stats saved to {output_path}")
    return baseline

def log_predictions(predictions_df: pd.DataFrame, batch_name: str = None):
    """
    Log predictions for monitoring.
    
    Args:
        predictions_df: DataFrame with customer_id, churn_probability, churn_prediction
        batch_name: Optional batch identifier
    """
    if batch_name is None:
        batch_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_path = MONITORING_DIR / f'predictions_{batch_name}.csv'
    predictions_df['logged_at'] = datetime.now().isoformat()
    predictions_df.to_csv(log_path, index=False)
    
    print(f"✓ Logged {len(predictions_df)} predictions to {log_path}")
    return log_path

def calculate_metrics(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame = None):
    """
    Calculate performance metrics on predictions.
    
    Args:
        predictions_df: DataFrame with predictions
        actuals_df: Optional DataFrame with actual outcomes
    
    Returns:
        dict: Performance metrics
    """
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'n_predictions': len(predictions_df),
        'avg_churn_probability': float(predictions_df['churn_probability'].mean()),
        'predicted_churn_count': int(predictions_df['churn_prediction'].sum()),
        'predicted_churn_rate': float(predictions_df['churn_prediction'].mean())
    }
    
    # If actuals provided, calculate accuracy
    if actuals_df is not None:
        merged = predictions_df.merge(actuals_df, on='customer_id', how='inner')
        if 'churn_actual' in merged.columns:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
            
            metrics['accuracy'] = float(accuracy_score(merged['churn_actual'], merged['churn_prediction']))
            metrics['precision'] = float(precision_score(merged['churn_actual'], merged['churn_prediction']))
            metrics['recall'] = float(recall_score(merged['churn_actual'], merged['churn_prediction']))
            
            if 'churn_probability' in merged.columns:
                metrics['roc_auc'] = float(roc_auc_score(merged['churn_actual'], merged['churn_probability']))
    
    return metrics

if __name__ == "__main__":
    # Generate baseline from cleaned dataset
    print("Generating baseline statistics...")
    df = pd.read_csv(OUTPUTS_DIR / 'cleaned_dataset.csv')
    baseline = generate_baseline_stats(df)
    
    print("\nBaseline Summary:")
    print(f"  Samples: {baseline['n_samples']:,}")
    print(f"  Features tracked: {len(baseline['feature_stats'])}")
    print(f"  Churn rate: {baseline['target_distribution']['churn_rate']*100:.2f}%")
