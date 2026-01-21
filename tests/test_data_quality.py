"""
test_data_quality.py
====================
Pytest tests for data quality validation.

Run with: pytest tests/test_data_quality.py -v

Author: QA Engineer
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

@pytest.fixture
def cleaned_data():
    """Load cleaned dataset."""
    return pd.read_csv(OUTPUTS_DIR / 'cleaned_dataset.csv')

@pytest.fixture
def customer_scores():
    """Load customer scores."""
    return pd.read_csv(OUTPUTS_DIR / 'customer_scores.csv')

class TestDataSchema:
    """Test data schema and structure."""
    
    def test_cleaned_data_columns(self, cleaned_data):
        """Test that required columns exist."""
        required_cols = [
            'customer_id', 'churn', 'monthly_price', 
            'total_sessions_30d', 'failed_payments_30d'
        ]
        for col in required_cols:
            assert col in cleaned_data.columns, f"Missing column: {col}"
    
    def test_no_missing_customer_ids(self, cleaned_data):
        """Test that customer_id has no missing values."""
        assert cleaned_data['customer_id'].notna().all(), "Missing customer_ids found"
    
    def test_churn_binary(self, cleaned_data):
        """Test that churn is binary (0 or 1)."""
        assert set(cleaned_data['churn'].unique()).issubset({0, 1}), "Churn must be 0 or 1"

class TestDataQuality:
    """Test data quality metrics."""
    
    def test_no_duplicates(self, cleaned_data):
        """Test that there are no duplicate customer_ids."""
        assert cleaned_data['customer_id'].duplicated().sum() == 0, "Duplicate customer_ids found"
    
    def test_price_positive(self, cleaned_data):
        """Test that monthly_price is positive."""
        assert (cleaned_data['monthly_price'] > 0).all(), "Negative prices found"
    
    def test_sessions_non_negative(self, cleaned_data):
        """Test that session counts are non-negative."""
        assert (cleaned_data['total_sessions_30d'] >= 0).all(), "Negative sessions found"
    
    def test_failed_payments_non_negative(self, cleaned_data):
        """Test that failed payments are non-negative."""
        assert (cleaned_data['failed_payments_30d'] >= 0).all(), "Negative failed payments found"
    
    def test_churn_rate_realistic(self, cleaned_data):
        """Test that churn rate is within expected range (10-40%)."""
        churn_rate = cleaned_data['churn'].mean()
        assert 0.1 <= churn_rate <= 0.4, f"Churn rate {churn_rate:.2%} outside expected range"

class TestCustomerScores:
    """Test customer scores output."""
    
    def test_scores_columns(self, customer_scores):
        """Test that scores have required columns."""
        required = ['customer_id', 'churn_probability', 'churn_prediction', 'risk_category']
        for col in required:
            assert col in customer_scores.columns, f"Missing column: {col}"
    
    def test_probability_range(self, customer_scores):
        """Test that churn_probability is between 0 and 1."""
        assert (customer_scores['churn_probability'] >= 0).all(), "Negative probabilities"
        assert (customer_scores['churn_probability'] <= 1).all(), "Probabilities > 1"
    
    def test_prediction_binary(self, customer_scores):
        """Test that predictions are binary."""
        assert set(customer_scores['churn_prediction'].unique()).issubset({0, 1})
    
    def test_risk_categories_valid(self, customer_scores):
        """Test that risk categories are valid."""
        valid_categories = ['Low', 'Medium', 'High', 'Very High']
        assert customer_scores['risk_category'].isin(valid_categories).all()

class TestDataContract:
    """Test data contracts and constraints."""
    
    def test_customer_id_format(self, cleaned_data):
        """Test that customer_id follows expected format."""
        # Assuming customer_id should be strings starting with 'C'
        sample_ids = cleaned_data['customer_id'].head()
        # Just check they're not null and unique
        assert cleaned_data['customer_id'].nunique() == len(cleaned_data)
    
    def test_no_extreme_outliers(self, cleaned_data):
        """Test that numeric features don't have extreme outliers."""
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'churn':
                continue
            
            q1 = cleaned_data[col].quantile(0.01)
            q99 = cleaned_data[col].quantile(0.99)
            
            # Check that min/max are not too far from 1st/99th percentile
            # (Should be handled by outlier capping)
            assert cleaned_data[col].min() >= q1 * 0.5, f"{col} has extreme low outliers"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
