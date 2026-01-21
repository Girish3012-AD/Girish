"""
main.py
=======
FastAPI Application for Customer Churn Prediction.

Endpoints:
- GET /health - Health check
- POST /predict - Single customer prediction
- POST /batch_predict - Batch predictions

Author: Senior ML Engineer
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from app.schemas import (
    CustomerInput, PredictionOutput,
    BatchCustomerInput, BatchPredictionOutput,
    HealthResponse
)
from app.utils import load_model, predict_single, predict_batch

# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    print("Loading model...")
    model = load_model()
    if model is not None:
        print("✓ Model loaded successfully!")
    else:
        print("⚠ Warning: Model not loaded!")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    API for predicting customer churn probability.
    
    ## Features
    - Single customer prediction
    - Batch prediction for multiple customers
    - Risk categorization (Low/Medium/High/Very High)
    
    ## Model
    The model is trained on customer behavior data including:
    - Demographics (age, location, device)
    - Subscription info (plan type, price)
    - Usage metrics (sessions, crashes)
    - Payment history (failed payments)
    - Support interactions (tickets, resolution time)
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and model loading status.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    Predict churn probability for a single customer.
    
    Returns:
    - churn_probability: Probability of churn (0-1)
    - churn_prediction: Binary prediction (0/1)
    - risk_category: Low/Medium/High/Very High
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic model to dict
        customer_data = customer.model_dump()
        
        # Make prediction
        result = predict_single(model, customer_data)
        
        return PredictionOutput(**result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionOutput, tags=["Prediction"])
async def batch_predict(batch: BatchCustomerInput):
    """
    Predict churn probability for multiple customers.
    
    Accepts a list of customer inputs and returns predictions for all.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if len(batch.customers) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty customer list provided."
        )
    
    if len(batch.customers) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 customers per batch request."
        )
    
    try:
        # Convert to list of dicts
        customers_data = [c.model_dump() for c in batch.customers]
        
        # Make batch predictions
        result = predict_batch(model, customers_data)
        
        # Convert predictions to Pydantic models
        predictions = [PredictionOutput(**p) for p in result["predictions"]]
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_customers=result["total_customers"],
            high_risk_count=result["high_risk_count"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
