"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class TransactionInput(BaseModel):
    """Single transaction input model."""
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str = "UGX"
    CountryCode: int = 256
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: str
    PricingStrategy: int
    FraudResult: int = 0
    
    @validator('TransactionStartTime')
    def validate_datetime(cls, v):
        """Validate datetime format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except:
            raise ValueError("TransactionStartTime must be in ISO format")
    
    @validator('FraudResult')
    def validate_fraud_result(cls, v):
        """Validate fraud result is 0 or 1."""
        if v not in [0, 1]:
            raise ValueError("FraudResult must be 0 or 1")
        return v


class PredictionRequest(BaseModel):
    """Request model for credit risk prediction."""
    transactions: List[TransactionInput] = Field(..., min_items=1, description="List of transactions")
    model_name: Optional[str] = Field("logistic_regression", description="Model to use for prediction")
    include_recommendations: Optional[bool] = Field(True, description="Include loan recommendations")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name."""
        valid_models = ['logistic_regression', 'xgboost', 'lightgbm']
        if v not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}")
        return v


class LoanRecommendation(BaseModel):
    """Loan recommendation model."""
    recommended_loan_amount: float = Field(..., description="Recommended loan amount")
    recommended_loan_duration_days: int = Field(..., description="Recommended loan duration in days")


class PredictionResult(BaseModel):
    """Single prediction result model."""
    CustomerId: str
    risk_probability: float = Field(..., ge=0, le=1, description="Risk probability (0-1)")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    risk_category: str = Field(..., description="Risk category: Low Risk, Medium Risk, or High Risk")
    loan_recommendation: Optional[LoanRecommendation] = None


class PredictionResponse(BaseModel):
    """Response model for credit risk prediction."""
    predictions: List[PredictionResult] = Field(..., description="List of predictions")
    model_used: str = Field(..., description="Model used for prediction")
    total_customers: int = Field(..., description="Total number of customers predicted")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="API status")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    version: str = Field("1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
