"""
retention_strategy.py
=====================
Retention Strategy Engine for Customer Churn Prevention.

Implements rule-based recommendations for retention actions
based on customer risk scores and segment profiles.

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import config
try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR


# =============================================================================
# RETENTION RULES
# =============================================================================

RETENTION_RULES = [
    {
        'name': 'Premium Discount + Call',
        'conditions': {
            'churn_probability': ('>=', 0.7),
            'monthly_price': ('>=', 500)
        },
        'action': 'Offer premium discount (20%) + Schedule retention call',
        'priority': 1
    },
    {
        'name': 'Payment Assistance',
        'conditions': {
            'churn_probability': ('>=', 0.6),
            'failed_payments_30d': ('>=', 1)
        },
        'action': 'Send payment reminder + Offer UPI cashback (5%)',
        'priority': 2
    },
    {
        'name': 'Priority Support',
        'conditions': {
            'churn_probability': ('>=', 0.5),
            'support_tickets_30d': ('>=', 2)
        },
        'action': 'Assign priority support + Issue apology credit (₹100)',
        'priority': 3
    },
    {
        'name': 'Re-engagement Campaign',
        'conditions': {
            'churn_probability': ('>=', 0.3),
            'total_sessions_30d': ('<=', 5)
        },
        'action': 'Send re-engagement push notification + Personalized content',
        'priority': 4
    },
    {
        'name': 'Loyalty Reward',
        'conditions': {
            'churn_probability': ('>=', 0.4),
            'churn_probability': ('<=', 0.6)
        },
        'action': 'Offer loyalty reward + Exclusive content access',
        'priority': 5
    },
    {
        'name': 'Downgrade Offer',
        'conditions': {
            'churn_probability': ('>=', 0.5),
            'plan_type': ('==', 'Premium')
        },
        'action': 'Offer temporary downgrade to Standard at same price',
        'priority': 6
    },
    {
        'name': 'Monitor Only',
        'conditions': {
            'churn_probability': ('<', 0.3)
        },
        'action': 'Continue monitoring - Low risk',
        'priority': 10
    }
]


# =============================================================================
# RULE ENGINE
# =============================================================================

def evaluate_condition(row: pd.Series, condition: Tuple) -> bool:
    """
    Evaluate a single condition against a row.
    
    Args:
        row: DataFrame row
        condition: Tuple of (operator, value)
    
    Returns:
        True if condition is met
    """
    operator, value = condition
    
    if operator == '>=':
        return row >= value
    elif operator == '<=':
        return row <= value
    elif operator == '>':
        return row > value
    elif operator == '<':
        return row < value
    elif operator == '==':
        return row == value
    elif operator == '!=':
        return row != value
    else:
        return False


def match_rule(row: pd.Series, rule: Dict) -> bool:
    """
    Check if a row matches all conditions in a rule.
    
    Args:
        row: DataFrame row
        rule: Rule dictionary with conditions
    
    Returns:
        True if all conditions are met
    """
    conditions = rule.get('conditions', {})
    
    for feature, condition in conditions.items():
        if feature not in row.index:
            continue
        
        if not evaluate_condition(row[feature], condition):
            return False
    
    return True


def get_recommended_action(row: pd.Series, rules: List[Dict] = None) -> str:
    """
    Get recommended retention action for a customer.
    
    Args:
        row: DataFrame row with customer data
        rules: List of retention rules
    
    Returns:
        Recommended action string
    """
    if rules is None:
        rules = RETENTION_RULES
    
    # Sort rules by priority
    sorted_rules = sorted(rules, key=lambda x: x.get('priority', 99))
    
    for rule in sorted_rules:
        if match_rule(row, rule):
            return rule['action']
    
    return 'No specific action - Continue standard engagement'


# =============================================================================
# RETENTION STRATEGY GENERATION
# =============================================================================

def generate_retention_actions(df: pd.DataFrame,
                                risk_scores: pd.DataFrame,
                                segments: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate retention actions for all customers.
    
    Args:
        df: Customer DataFrame with features
        risk_scores: DataFrame with churn probabilities
        segments: Optional DataFrame with segment assignments
    
    Returns:
        DataFrame with customer_id, churn_probability, segment_id, recommended_action
    """
    # Merge data
    merged = df.copy()
    
    if 'customer_id' in risk_scores.columns:
        merged = merged.merge(
            risk_scores[['customer_id', 'churn_probability']], 
            on='customer_id', 
            how='left'
        )
    else:
        merged['churn_probability'] = 0.5
    
    if segments is not None and 'customer_id' in segments.columns:
        merged = merged.merge(
            segments[['customer_id', 'segment_id']], 
            on='customer_id', 
            how='left'
        )
    else:
        merged['segment_id'] = 0
    
    # Generate recommendations
    print("Generating retention recommendations...")
    
    recommendations = []
    for idx, row in merged.iterrows():
        action = get_recommended_action(row)
        recommendations.append(action)
    
    merged['recommended_action'] = recommendations
    
    # Create output DataFrame
    output_cols = ['customer_id', 'churn_probability', 'segment_id', 'recommended_action']
    available_cols = [c for c in output_cols if c in merged.columns]
    
    result = merged[available_cols].copy()
    
    return result


def summarize_retention_actions(actions_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for retention actions.
    
    Args:
        actions_df: DataFrame with retention actions
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # Action distribution
    action_counts = actions_df['recommended_action'].value_counts()
    summary['action_distribution'] = action_counts.to_dict()
    
    # Segment distribution
    if 'segment_id' in actions_df.columns:
        segment_counts = actions_df['segment_id'].value_counts()
        summary['segment_distribution'] = segment_counts.to_dict()
    
    # Risk distribution
    if 'churn_probability' in actions_df.columns:
        summary['avg_churn_probability'] = actions_df['churn_probability'].mean()
        summary['high_risk_count'] = (actions_df['churn_probability'] >= 0.7).sum()
        summary['medium_risk_count'] = (
            (actions_df['churn_probability'] >= 0.3) & 
            (actions_df['churn_probability'] < 0.7)
        ).sum()
        summary['low_risk_count'] = (actions_df['churn_probability'] < 0.3).sum()
    
    return summary


def print_retention_summary(actions_df: pd.DataFrame, summary: Dict) -> None:
    """
    Print summary of retention recommendations.
    
    Args:
        actions_df: DataFrame with retention actions
        summary: Summary dictionary
    """
    print("\n" + "=" * 60)
    print("RETENTION STRATEGY SUMMARY")
    print("=" * 60)
    
    # Risk distribution
    if 'high_risk_count' in summary:
        print(f"\n📊 Risk Distribution:")
        print(f"   High Risk (>=70%): {summary['high_risk_count']:,}")
        print(f"   Medium Risk (30-70%): {summary['medium_risk_count']:,}")
        print(f"   Low Risk (<30%): {summary['low_risk_count']:,}")
    
    # Segment distribution
    if 'segment_distribution' in summary:
        print(f"\n🎯 Customers per Segment:")
        for segment, count in summary['segment_distribution'].items():
            print(f"   Segment {segment}: {count:,}")
    
    # Top actions
    print(f"\n💡 Top 3 Recommended Actions:")
    for action, count in list(summary.get('action_distribution', {}).items())[:3]:
        print(f"   • {action}: {count:,} customers")
    
    # Top 20 high-risk customers
    if 'churn_probability' in actions_df.columns:
        print(f"\n🚨 Top 20 High-Risk Customers:")
        top_risk = actions_df.nlargest(20, 'churn_probability')
        for idx, row in top_risk.iterrows():
            print(f"   {row.get('customer_id', 'Unknown')}: "
                  f"{row['churn_probability']*100:.1f}% - {row['recommended_action'][:50]}")


def save_retention_actions(actions_df: pd.DataFrame, 
                           filepath: Path = None) -> Path:
    """Save retention actions to CSV."""
    if filepath is None:
        filepath = OUTPUT_DIR / 'retention_actions.csv'
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    actions_df.to_csv(filepath, index=False)
    print(f"✓ Saved retention actions: {filepath}")
    
    return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Retention Strategy module loaded successfully.")
    print(f"\nConfigured retention rules: {len(RETENTION_RULES)}")
    for rule in RETENTION_RULES:
        print(f"  • {rule['name']}: {rule['action'][:50]}...")
