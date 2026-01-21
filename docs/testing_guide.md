# Testing Guide

## Overview

This project includes comprehensive automated tests using **Pytest** to ensure data quality, API functionality, and model correctness.

---

## Test Structure

```
tests/
├── test_data_quality.py      # Data validation tests
├── test_api.py                # FastAPI endpoint tests
└── test_model_inference.py    # Model prediction tests
```

---

## Installation

Install test dependencies:

```bash
pip install pytest requests
```

---

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_data_quality.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov=app --cov-report=html
```

---

## Test Categories

### 1. Data Quality Tests (`test_data_quality.py`)

**Purpose**: Validate data schema, quality, and business rules

**Test Classes**:

#### `TestDataSchema`
- `test_cleaned_data_columns`: Required columns exist
- `test_no_missing_customer_ids`: No null customer IDs
- `test_churn_binary`: Churn is 0 or 1

#### `TestDataQuality`
- `test_no_duplicates`: No duplicate customer IDs
- `test_price_positive`: Monthly price > 0
- `test_sessions_non_negative`: Sessions >= 0
- `test_failed_payments_non_negative`: Failed payments >= 0
- `test_churn_rate_realistic`: Churn rate in 10-40% range

#### `TestCustomerScores`
- `test_scores_columns`: Scores have required fields
- `test_probability_range`: Churn probability in [0, 1]
- `test_prediction_binary`: Predictions are 0 or 1
- `test_risk_categories_valid`: Risk categories are valid

#### `TestDataContract`
- `test_customer_id_format`: IDs follow expected format
- `test_no_extreme_outliers`: No extreme outliers after capping

**Run**:
```bash
pytest tests/test_data_quality.py -v
```

---

### 2. API Tests (`test_api.py`)

**Purpose**: Validate FastAPI endpoints

⚠️ **Prerequisites**: API must be running on `http://localhost:8000`

```bash
# Terminal 1: Start API
uvicorn app.main:app --reload

# Terminal 2: Run tests
pytest tests/test_api.py -v
```

**Test Classes**:

#### `TestHealthEndpoint`
- `test_health_check`: Health endpoint returns 200 + healthy status

#### `TestPredictEndpoint`
- `test_predict_success`: Single prediction works
- `test_predict_probability_range`: Probability in [0, 1]
- `test_predict_missing_field`: Returns 422 for missing fields

#### `TestBatchPredictEndpoint`
- `test_batch_predict_success`: Batch predictions work
- `test_batch_predict_empty`: Handles empty batch

#### `TestErrorHandling`
- `test_invalid_json`: Returns 422 for invalid JSON
- `test_negative_values`: Returns 422 for invalid values

---

### 3. Model Inference Tests (`test_model_inference.py`)

**Purpose**: Validate model predictions and performance

**Test Classes**:

#### `TestModelLoading`
- `test_model_exists`: Model file exists
- `test_model_loads`: Model can be loaded

#### `TestModelInference`
- `test_predict_shape`: Predictions have correct shape
- `test_predict_proba_shape`: Probabilities have correct shape
- `test_predict_binary`: Predictions are 0 or 1
- `test_predict_proba_range`: Probabilities in [0, 1]
- `test_predict_proba_sum_to_one`: Probabilities sum to 1

#### `TestFeatureEngineering`
- `test_interaction_features`: Interaction features created correctly
- `test_log_transforms`: Log transforms applied correctly

#### `TestModelPerformance`
- `test_model_accuracy_threshold`: Accuracy >= 70%
- `test_model_roc_auc_threshold`: ROC-AUC >= 0.75

**Run**:
```bash
pytest tests/test_model_inference.py -v
```

---

## Data Contract Validation

**File**: `src/data_contract.py`

**Purpose**: Define and enforce data schemas, constraints, and business rules

### Run Validation
```bash
python src/data_contract.py
```

**Output**: `outputs/data_contract_report.txt`

### Contract Definition

**Schema Constraints**:
- Column types (int, float, categorical, string)
- Nullable constraints
- Min/max values
- Allowed values (for categorical)
- Uniqueness

**Business Rules**:
- Churn rate in 10-40% range
- Premium plans cost more than Standard/Basic
- Custom validation functions

### Example Usage

```python
from src.data_contract import validate_schema, generate_contract_report
import pandas as pd

df = pd.read_csv('outputs/cleaned_dataset.csv')

# Validate
results = validate_schema(df)

if results['valid']:
    print("✓ Data contract passed")
else:
    print("❌ Validation errors:")
    for error in results['errors']:
        print(f"  - {error}")

# Generate full report
is_valid = generate_contract_report(df,'outputs/contract_report.txt')
```

---

## CI/CD Integration

### GitHub Actions (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run data quality tests
      run: pytest tests/test_data_quality.py -v
    
    - name: Run model tests
      run: pytest tests/test_model_inference.py -v
    
    - name: Data contract validation
      run: python src/data_contract.py
```

---

## Test Coverage Goals

| Component | Target Coverage |
|-----------|----------------|
| Data Pipeline | 80% |
| Feature Engineering | 90% |
| Model Training | 70% |
| API | 85% |
| Utilities | 75% |

---

## Continuous Testing

### Pre-Commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
pytest tests/test_data_quality.py -q
if [ $? -ne 0 ]; then
    echo "❌ Data quality tests failed. Commit aborted."
    exit 1
fi
```

### Daily Monitoring

Run tests daily against production data:

```bash
# Cron job: Daily at 2 AM
0 2 * * * cd /path/to/churn_project && pytest tests/ --junitxml=test-results.xml
```

---

## Troubleshooting

### API Tests Failing

**Error**: `Connection refused`

**Solution**: Ensure API is running:
```bash
uvicorn app.main:app --reload
```

### Model Tests Failing

**Error**: `FileNotFoundError: best_model.pkl`

**Solution**: Train model first:
```bash
python src/train_models.py
```

### Data Quality Tests Failing

**Error**: `File not found: cleaned_dataset.csv`

**Solution**: Run data pipeline:
```bash
python src/build_final_dataset.py
python src/preprocess.py
```

---

## Best Practices

1. **Run tests before committing**
2. **Update tests when adding features**
3. **Use fixtures to avoid code duplication**
4. **Mock external dependencies** (e.g., `@pytest.mark.mock`)
5. **Test edge cases** (empty inputs, extreme values)
6. **Maintain >= 80% coverage** for critical modules

---

## Next Steps

1. **Add integration tests**: End-to-end workflow tests
2. **Performance tests**: Measure API latency
3. **Load tests**: Test API under high traffic (Locust)
4. **Security tests**: SQL injection, XSS prevention
5. **Regression tests**: Compare new model vs. baseline
