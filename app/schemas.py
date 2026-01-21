"""
schemas.py
==========
Pydantic schemas for FastAPI churn prediction API.

Author: Senior ML Engineer
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class CustomerInput(BaseModel):
    """Input schema for single customer prediction."""
    
    age: int = Field(..., ge=10, le=100, description="Customer age (10-100)")
    gender: str = Field(..., description="Gender: Male/Female/Other")
    location: str = Field(..., description="City name")
    device_type: str = Field(..., description="Android/iOS/Web")
    acquisition_channel: str = Field(..., description="Organic/Ads/Referral/Partner")
    plan_type: str = Field(..., description="Basic/Standard/Premium")
    monthly_price: float = Field(..., ge=0, description="Monthly subscription price")
    auto_renew: int = Field(..., ge=0, le=1, description="Auto-renewal status (0/1)")
    total_sessions_30d: float = Field(0, ge=0, description="Sessions in last 30 days")
    avg_session_minutes_30d: float = Field(0, ge=0, description="Avg session duration")
    total_crashes_30d: float = Field(0, ge=0, description="Crashes in last 30 days")
    failed_payments_30d: float = Field(0, ge=0, description="Failed payments count")
    total_amount_success_30d: float = Field(0, ge=0, description="Successful payment amount")
    support_tickets_30d: float = Field(0, ge=0, description="Support tickets count")
    avg_resolution_time_30d: float = Field(0, ge=0, description="Avg ticket resolution time")
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for single prediction."""
    
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: int = Field(..., description="Binary prediction (0/1)")
    risk_category: str = Field(..., description="Low/Medium/High/Very High")
    
    class Config:
        json_schema_extra = {
            "example": {
                "churn_probability": 0.73,
                "churn_prediction": 1,
                "risk_category": "Very High"
            }
        }


class BatchCustomerInput(BaseModel):
    """Input schema for batch predictions."""
    
    customers: List[CustomerInput] = Field(..., description="List of customer inputs")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")
    total_customers: int = Field(..., description="Total customers processed")
    high_risk_count: int = Field(..., description="Number of high-risk customers")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
