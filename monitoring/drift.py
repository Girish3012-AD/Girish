"""
drift.py
========
Data drift detection using Population Stability Index (PSI).

PSI Formula:
PSI = Σ (actual% - expected%) × ln(actual% / expected%)

Interpretation:
- PSI < 0.1: No significant change
- 0.1 <= PSI < 0.2: Moderate change
- PSI >= 0.2: Significant change (retraining recommended)

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

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10):
    """
    Calculate Population Stability Index (PSI).
    
    Args:
        expected: Baseline feature values
        actual: Current feature values
        bins: Number of bins for discretization
    
    Returns:
        float: PSI value
    """
    # Create bins from expected distribution
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Bin both distributions
    expected_binned = np.histogram(expected, bins=breakpoints)[0]
    actual_binned = np.histogram(actual, bins=breakpoints)[0]
    
    # Convert to percentages (add small epsilon to avoid division by zero)
    epsilon = 1e-5
    expected_pct = expected_binned / len(expected) + epsilon
    actual_pct = actual_binned / len(actual) + epsilon
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)

def detect_drift(baseline_path: Path, current_df: pd.DataFrame, threshold: float = 0.2):
    """
    Detect drift by comparing current data to baseline.
    
    Args:
        baseline_path: Path to baseline_stats.json
        current_df: Current data to check
        threshold: PSI threshold for drift alert
    
    Returns:
        dict: Drift report
    """
    # Load baseline
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    drift_report = {
        'timestamp': datetime.now().isoformat(),
        'baseline_date': baseline['created_at'],
        'n_current_samples': len(current_df),
        'n_baseline_samples': baseline['n_samples'],
        'feature_drift': {},
        'drifted_features': [],
        'overall_drift_detected': False
    }
    
    # Check each feature
    for feature, stats in baseline['feature_stats'].items():
        if feature not in current_df.columns:
            continue
        
        # Calculate PSI
        # Use baseline stats to create synthetic baseline distribution
        baseline_mean = stats['mean']
        baseline_std = stats['std']
        baseline_samples = np.random.normal(baseline_mean, baseline_std, baseline['n_samples'])
        
        current_samples = current_df[feature].dropna().values
        
        if len(current_samples) == 0:
            continue
        
        psi = calculate_psi(baseline_samples, current_samples)
        
        drift_report['feature_drift'][feature] = {
            'psi': psi,
            'drifted': psi >= threshold,
            'current_mean': float(current_df[feature].mean()),
            'baseline_mean': baseline_mean,
            'mean_shift': float(current_df[feature].mean() - baseline_mean)
        }
        
        if psi >= threshold:
            drift_report['drifted_features'].append(feature)
    
    drift_report['overall_drift_detected'] = len(drift_report['drifted_features']) > 0
    
    return drift_report

def generate_drift_report(drift_data: dict, output_path: Path = None):
    """
    Generate human-readable drift report.
    
    Args:
        drift_data: Drift detection results
        output_path: Where to save report
    """
    if output_path is None:
        output_path = MONITORING_DIR / 'drift_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DATA DRIFT DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Report Generated: {drift_data['timestamp']}\n")
        f.write(f"Baseline Date: {drift_data['baseline_date']}\n")
        f.write(f"Current Samples: {drift_data['n_current_samples']:,}\n")
        f.write(f"Baseline Samples: {drift_data['n_baseline_samples']:,}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("DRIFT STATUS\n")
        f.write("=" * 60 + "\n\n")
        
        if drift_data['overall_drift_detected']:
            f.write(f"⚠️  DRIFT DETECTED in {len(drift_data['drifted_features'])} feature(s)\n\n")
            f.write("Drifted Features:\n")
            for feature in drift_data['drifted_features']:
                psi = drift_data['feature_drift'][feature]['psi']
                f.write(f"  - {feature}: PSI = {psi:.4f}\n")
        else:
            f.write("✓ No significant drift detected\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("FEATURE-LEVEL PSI SCORES\n")
        f.write("=" * 60 + "\n\n")
        
        for feature, metrics in sorted(drift_data['feature_drift'].items(), 
                                       key=lambda x: x[1]['psi'], reverse=True):
            psi = metrics['psi']
            status = "⚠️ DRIFT" if metrics['drifted'] else "✓ OK"
            f.write(f"{feature:30s} PSI={psi:6.4f}  {status}\n")
            f.write(f"  Current Mean: {metrics['current_mean']:10.2f}\n")
            f.write(f"  Baseline Mean: {metrics['baseline_mean']:10.2f}\n")
            f.write(f"  Shift: {metrics['mean_shift']:+10.2f}\n\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 60 + "\n\n")
        f.write("PSI < 0.1:  No significant change\n")
        f.write("0.1 <= PSI < 0.2:  Moderate change\n")
        f.write("PSI >= 0.2:  Significant change (retraining recommended)\n")
    
    print(f"✓ Drift report saved to {output_path}")

if __name__ == "__main__":
    # Example: Detect drift on current data
    baseline_path = MONITORING_DIR / 'baseline_stats.json'
    
    if not baseline_path.exists():
        print("❌ Baseline not found. Run monitoring/monitor.py first.")
    else:
        # Load current data (use cleaned dataset as example)
        from pathlib import Path
        current_df = pd.read_csv(PROJECT_ROOT / 'outputs' / 'cleaned_dataset.csv')
        
        print("Detecting drift...")
        drift_data = detect_drift(baseline_path, current_df)
        
        # Generate report
        generate_drift_report(drift_data)
        
        # Save JSON
        with open(MONITORING_DIR / 'drift_results.json', 'w') as f:
            json.dump(drift_data, f, indent=2)
        
        print("\nDrift Detection Summary:")
        if drift_data['overall_drift_detected']:
            print(f"  ⚠️  Drift detected in {len(drift_data['drifted_features'])} features")
        else:
            print("  ✓ No significant drift")
