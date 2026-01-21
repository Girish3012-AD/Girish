"""
test_model_inference.py
=======================
Pytest tests for model inference and predictions.

Run with: pytest tests/test_model_inference.py -v

Author: QA Engineer
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'outputs' / 'models' / 'best_model.pkl'
CLEANED_DATA_PATH = PROJECT_ROOT / 'outputs' / 'cleaned_dataset.csv'

@pytest.fixture
def model():
    """Load trained model."""
    return joblib.load(MODEL_PATH)

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    return df.drop(columns=['churn']).head(10)

class TestModelLoading:
    """Test model loading."""
    
    def test_model_exists(self):
        """Test that model file exists."""
        assert MODEL_PATH.exists(), "Model file not found"
    
    def test_model_loads(self, model):
        """Test that model can be loaded."""
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

class TestModelInference:
    """Test model predictions."""
    
    def test_predict_shape(self, model, sample_data):
        """Test that predictions have correct shape."""
        predictions = model.predict(sample_data)
        assert len(predictions) == len(sample_data)
    
    def test_predict_proba_shape(self, model, sample_data):
        """Test that probability predictions have correct shape."""
        probas = model.predict_proba(sample_data)
        assert probas.shape == (len(sample_data), 2)
    
    def test_predict_binary(self, model, sample_data):
        """Test that predictions are binary (0 or 1)."""
        predictions = model.predict(sample_data)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba_range(self, model, sample_data):
        """Test that probabilities are between 0 and 1."""
        probas = model.predict_proba(sample_data)
        assert (probas >= 0).all() and (probas <= 1).all()
    
    def test_predict_proba_sum_to_one(self, model, sample_data):
        """Test that probabilities sum to 1."""
        probas = model.predict_proba(sample_data)
        sums = probas.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(len(sample_data)), decimal=5)

class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_interaction_features(self):
        """Test that interaction features are created correctly."""
        from src.features import create_interaction_features
        
        df = pd.DataFrame({
            'total_sessions_30d': [10, 20, 30],
            'total_crashes_30d': [1, 0, 2],
            'failed_payments_30d': [0, 1, 2],
            'total_amount_success_30d': [100, 200, 300]
        })
        
        df_new = create_interaction_features(df.copy())
        
        # Check new columns exist
        assert 'sessions_per_crash' in df_new.columns
        assert 'payment_failure_rate' in df_new.columns
    
    def test_log_transforms(self):
        """Test log transforms."""
        from src.features import apply_log_transforms
        
        df = pd.DataFrame({
            'total_sessions_30d': [10, 20, 30],
            'total_amount_success_30d': [100, 200, 300]
        })
        
        df_new = apply_log_transforms(df.copy())
        
        assert 'log_sessions' in df_new.columns
        assert 'log_amount' in df_new.columns
        assert (df_new['log_sessions'] >= 0).all()

class TestModelPerformance:
    """Test model performance metrics."""
    
    def test_model_accuracy_threshold(self, model):
        """Test that model meets minimum accuracy threshold."""
        df = pd.read_csv(CLEANED_DATA_PATH)
        X = df.drop(columns=['churn'])
        y = df['churn']
        
        predictions = model.predict(X)
        accuracy = (predictions == y).mean()
        
        # Expect at least 70% accuracy
        assert accuracy >= 0.70, f"Model accuracy {accuracy:.2%} below threshold"
    
    def test_model_roc_auc_threshold(self, model):
        """Test that model meets minimum ROC-AUC threshold."""
        from sklearn.metrics import roc_auc_score
        
        df = pd.read_csv(CLEANED_DATA_PATH)
        X = df.drop(columns=['churn'])
        y = df['churn']
        
        probas = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, probas)
        
        # Expect at least 0.75 ROC-AUC
        assert roc_auc >= 0.75, f"Model ROC-AUC {roc_auc:.3f} below threshold"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
