# Customer Churn Prediction API

FastAPI-based REST API for real-time churn prediction.

## Quick Start

```bash
# Install dependencies
pip install -r requirements_api.txt

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/predict` | Single prediction |
| POST | `/batch_predict` | Batch predictions |

## Sample Requests

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "location": "Mumbai",
    "device_type": "Android",
    "acquisition_channel": "Organic",
    "plan_type": "Standard",
    "monthly_price": 499.0,
    "auto_renew": 1,
    "total_sessions_30d": 45,
    "avg_session_minutes_30d": 25.5,
    "total_crashes_30d": 2,
    "failed_payments_30d": 0,
    "total_amount_success_30d": 499.0,
    "support_tickets_30d": 1,
    "avg_resolution_time_30d": 12.5
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"customers": [<customer1>, <customer2>, ...]}'
```

## Response Format

```json
{
  "churn_probability": 0.73,
  "churn_prediction": 1,
  "risk_category": "Very High"
}
```

## Risk Categories

| Probability | Category |
|-------------|----------|
| >= 0.7 | Very High |
| >= 0.5 | High |
| >= 0.3 | Medium |
| < 0.3 | Low |
