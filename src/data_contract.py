"""
data_contract.py
================
Data contract definitions and validation for churn prediction.

Defines expected schemas, constraints, and validation rules.

Author: Data Engineer
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Data Contract Definition
DATA_CONTRACT = {
    "version": "1.0",
    "dataset_name": "customer_churn",
    "description": "Customer churn prediction dataset",
    
    "schema": {
        "customer_id": {
            "type": "string",
            "nullable": False,
            "unique": True,
            "description": "Unique customer identifier"
        },
        "plan_type": {
            "type": "categorical",
            "nullable": False,
            "allowed_values": ["Basic", "Standard", "Premium"],
            "description": "Subscription plan type"
        },
        "monthly_price": {
            "type": "float",
            "nullable": False,
            "min": 0,
            "max": 10000,
            "description": "Monthly subscription price in INR"
        },
        "age": {
            "type": "int",
            "nullable": False,
            "min": 18,
            "max": 100,
            "description": "Customer age"
        },
        "gender": {
            "type": "categorical",
            "nullable": False,
            "allowed_values": ["Male", "Female"],
            "description": "Customer gender"
        },
        "location": {
            "type": "categorical",
            "nullable": False,
            "description": "Customer city/location"
        },
        "device_type": {
            "type": "categorical",
            "nullable": False,
            "allowed_values": ["Android", "iOS", "Web"],
            "description": "Primary device type"
        },
        "acquisition_channel": {
            "type": "categorical",
            "nullable": False,
            "allowed_values": ["Organic", "Paid Ads", "Referral"],
            "description": "How customer was acquired"
        },
        "auto_renew": {
            "type": "int",
            "nullable": False,
            "allowed_values": [0, 1],
            "description": "Auto-renewal enabled flag"
        },
        "total_sessions_30d": {
            "type": "int",
            "nullable": False,
            "min": 0,
            "max": 1000,
            "description": "Total app sessions in last 30 days"
        },
        "avg_session_minutes_30d": {
            "type": "float",
            "nullable": False,
            "min": 0,
            "max": 300,
            "description": "Average session duration in minutes"
        },
        "total_crashes_30d": {
            "type": "int",
            "nullable": False,
            "min": 0,
            "max": 100,
            "description": "Total app crashes in last 30 days"
        },
        "failed_payments_30d": {
            "type": "int",
            "nullable": False,
            "min": 0,
            "max": 10,
            "description": "Failed payment attempts in last 30 days"
        },
        "total_amount_success_30d": {
            "type": "float",
            "nullable": False,
            "min": 0,
            "max": 50000,
            "description": "Total successful payment amount in INR"
        },
        "support_tickets_30d": {
            "type": "int",
            "nullable": False,
            "min": 0,
            "max": 50,
            "description": "Support tickets raised in last 30 days"
        },
        "avg_resolution_time_30d": {
            "type": "float",
            "nullable": False,
            "min": 0,
            "max": 240,
            "description": "Average ticket resolution time in hours"
        },
        "churn": {
            "type": "int",
            "nullable": False,
            "allowed_values": [0, 1],
            "description": "Churn label (1=churned, 0=retained)"
        }
    },
    
    "business_rules": [
        {
            "rule": "churn_rate_range",
            "description": "Overall churn rate should be between 10% and 40%",
            "check": lambda df: 0.10 <= df['churn'].mean() <= 0.40
        },
        {
            "rule": "premium_pricing",
            "description": "Premium plans should cost more than Standard and Basic",
            "check": lambda df: df[df['plan_type'] == 'Premium']['monthly_price'].mean() > 
                               df[df['plan_type'] == 'Standard']['monthly_price'].mean()
        }
    ]
}

def validate_schema(df: pd.DataFrame, contract: Dict = DATA_CONTRACT) -> Dict[str, Any]:
    """
    Validate DataFrame against data contract schema.
    
    Args:
        df: DataFrame to validate
        contract: Data contract definition
    
    Returns:
        dict: Validation results with errors and warnings
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    schema = contract["schema"]
    
    # Check required columns
    for col, spec in schema.items():
        if col not in df.columns:
            results["errors"].append(f"Missing required column: {col}")
            results["valid"] = False
            continue
        
        # Check nullable
        if not spec.get("nullable", True) and df[col].isna().any():
            results["errors"].append(f"Column '{col}' has null values but is not nullable")
            results["valid"] = False
        
        # Check type
        if spec["type"] == "int":
            if not pd.api.types.is_integer_dtype(df[col]):
                results["warnings"].append(f"Column '{col}' should be int, got {df[col].dtype}")
        
        elif spec["type"] == "float":
            if not pd.api.types.is_numeric_dtype(df[col]):
                results["errors"].append(f"Column '{col}' should be numeric, got {df[col].dtype}")
                results["valid"] = False
        
        # Check min/max
        if "min" in spec:
            if (df[col] < spec["min"]).any():
                results["errors"].append(f"Column '{col}' has values below min={spec['min']}")
                results["valid"] = False
        
        if "max" in spec:
            if (df[col] > spec["max"]).any():
                results["errors"].append(f"Column '{col}' has values above max={spec['max']}")
                results["valid"] = False
        
        # Check allowed values
        if "allowed_values" in spec:
            invalid = ~df[col].isin(spec["allowed_values"] + [np.nan] if spec.get("nullable") else spec["allowed_values"])
            if invalid.any():
                invalid_vals = df.loc[invalid, col].unique()
                results["errors"].append(
                    f"Column '{col}' has invalid values: {invalid_vals.tolist()}"
                )
                results["valid"] = False
        
        # Check uniqueness
        if spec.get("unique", False):
            if df[col].duplicated().any():
                results["errors"].append(f"Column '{col}' should be unique but has duplicates")
                results["valid"] = False
    
    return results

def validate_business_rules(df: pd.DataFrame, contract: Dict = DATA_CONTRACT) -> Dict[str, Any]:
    """
    Validate business rules.
    
    Args:
        df: DataFrame to validate
        contract: Data contract with business rules
    
    Returns:
        dict: Validation results
    """
    results = {
        "valid": True,
        "rule_results": []
    }
    
    for rule in contract.get("business_rules", []):
        try:
            passed = rule["check"](df)
            results["rule_results"].append({
                "rule": rule["rule"],
                "description": rule["description"],
                "passed": passed
            })
            if not passed:
                results["valid"] = False
        except Exception as e:
            results["rule_results"].append({
                "rule": rule["rule"],
                "description": rule["description"],
                "passed": False,
                "error": str(e)
            })
            results["valid"] = False
    
    return results

def generate_contract_report(df: pd.DataFrame, output_path: str = None):
    """
    Generate full data contract validation report.
    
    Args:
        df: DataFrame to validate
        output_path: Optional path to save report
    """
    schema_results = validate_schema(df)
    business_results = validate_business_rules(df)
    
    report = []
    report.append("=" * 60)
    report.append("DATA CONTRACT VALIDATION REPORT")
    report.append("=" * 60)
    report.append(f"\nDataset: {DATA_CONTRACT['dataset_name']}")
    report.append(f"Version: {DATA_CONTRACT['version']}")
    report.append(f"Rows: {len(df):,}")
    report.append(f"Columns: {len(df.columns)}")
    
    report.append("\n" + "=" * 60)
    report.append("SCHEMA VALIDATION")
    report.append("=" * 60)
    
    if schema_results["valid"]:
        report.append("\n✓ Schema validation PASSED")
    else:
        report.append("\n❌ Schema validation FAILED")
        report.append("\nErrors:")
        for error in schema_results["errors"]:
            report.append(f"  - {error}")
    
    if schema_results["warnings"]:
        report.append("\nWarnings:")
        for warning in schema_results["warnings"]:
            report.append(f"  - {warning}")
    
    report.append("\n" + "=" * 60)
    report.append("BUSINESS RULES VALIDATION")
    report.append("=" * 60)
    
    for rule_result in business_results["rule_results"]:
        status = "✓ PASS" if rule_result["passed"] else "❌ FAIL"
        report.append(f"\n{status}: {rule_result['rule']}")
        report.append(f"  {rule_result['description']}")
        if "error" in rule_result:
            report.append(f"  Error: {rule_result['error']}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Report saved to {output_path}")
    
    return schema_results["valid"] and business_results["valid"]

if __name__ == "__main__":
    # Example validation
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent
    df = pd.read_csv(PROJECT_ROOT / 'outputs' / 'cleaned_dataset.csv')
    
    is_valid = generate_contract_report(df, PROJECT_ROOT / 'outputs' / 'data_contract_report.txt')
    
    if is_valid:
        print("\n✓ All validations passed!")
    else:
        print("\n❌ Some validations failed. Check report for details.")
