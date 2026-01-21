"""
test_api.py
===========
Pytest tests for FastAPI endpoints.

Run with: pytest tests/test_api.py -v

Note: Requires API to be running on http://localhost:8000

Author: QA Engineer
"""

import pytest
import requests
import json

BASE_URL = "http://localhost:8000"

@pytest.fixture
def sample_customer():
    """Sample customer data for testing."""
    return {
        "customer_id": "TEST_001",
        "plan_type": "Premium",
        "monthly_price": 999.0,
        "age": 28,
        "gender": "Male",
        "location": "Mumbai",
        "device_type": "Android",
        "acquisition_channel": "Organic",
        "auto_renew": 1,
        "total_sessions_30d": 45,
        "avg_session_minutes_30d": 25.5,
        "total_crashes_30d": 2,
        "failed_payments_30d": 0,
        "total_amount_success_30d": 999.0,
        "support_tickets_30d": 0,
        "avg_resolution_time_30d": 0.0
    }

class TestHealthEndpoint:
    """Test /health endpoint."""
    
    def test_health_check(self):
        """Test that health endpoint returns 200."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data
        assert data['model_loaded'] is True

class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    def test_predict_success(self, sample_customer):
        """Test successful prediction."""
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_customer
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'customer_id' in data
        assert 'churn_probability' in data
        assert 'churn_prediction' in data
        assert 'risk_category' in data
        
        # Validate types
        assert isinstance(data['churn_probability'], float)
        assert isinstance(data['churn_prediction'], int)
        assert data['churn_prediction'] in [0, 1]
    
    def test_predict_probability_range(self, sample_customer):
        """Test that probability is between 0 and 1."""
        response = requests.post(f"{BASE_URL}/predict", json=sample_customer)
        data = response.json()
        
        assert 0 <= data['churn_probability'] <= 1
    
    def test_predict_missing_field(self):
        """Test with missing required field."""
        incomplete_data = {"customer_id": "TEST_002", "age": 30}
        
        response = requests.post(f"{BASE_URL}/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error

class TestBatchPredictEndpoint:
    """Test /batch_predict endpoint."""
    
    def test_batch_predict_success(self, sample_customer):
        """Test batch prediction with multiple customers."""
        batch_data = {
            "customers": [
                sample_customer,
                {**sample_customer, "customer_id": "TEST_002", "age": 35}
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=batch_data
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'predictions' in data
        assert len(data['predictions']) == 2
        assert 'total_customers' in data
        assert data['total_customers'] == 2
    
    def test_batch_predict_empty(self):
        """Test batch predict with empty list."""
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json={"customers": []}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_customers'] == 0

class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json(self):
        """Test with invalid JSON."""
        response = requests.post(
            f"{BASE_URL}/predict",
            data="invalid json",
            headers={'Content-Type': 'application/json'}
        )
        assert response.status_code == 422
    
    def test_negative_values(self, sample_customer):
        """Test with negative values that should fail validation."""
        invalid_data = {**sample_customer, "age": -5}
        
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
