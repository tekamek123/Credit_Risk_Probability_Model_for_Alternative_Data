"""
FastAPI application for credit risk model inference.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.predict import CreditRiskPredictor
from src.api.pydantic_models import (
    PredictionRequest, PredictionResponse, PredictionResult,
    LoanRecommendation, HealthResponse, ErrorResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for credit risk probability prediction and loan recommendations",
    version="1.0.0"
)

# Initialize predictor (lazy loading)
predictor: CreditRiskPredictor = None


def get_predictor() -> CreditRiskPredictor:
    """Get or initialize predictor (singleton pattern)."""
    global predictor
    if predictor is None:
        predictor = CreditRiskPredictor(model_dir="models")
        try:
            predictor.load_models('logistic_regression')
        except FileNotFoundError:
            # Models not trained yet
            pass
    return predictor


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    pred = get_predictor()
    models_loaded = list(pred.models.keys()) if pred.models else []
    
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        models_loaded=models_loaded,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return await root()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict credit risk for given transactions.
    
    Args:
        request: Prediction request with transactions
        
    Returns:
        Prediction response with risk probabilities, credit scores, and recommendations
    """
    try:
        # Get predictor
        pred = get_predictor()
        
        # Check if model is loaded
        if request.model_name not in pred.models:
            try:
                pred.load_models(request.model_name)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model_name}' not found. Please train the model first."
                )
        
        # Convert transactions to DataFrame
        transactions_data = [t.dict() for t in request.transactions]
        df = pd.DataFrame(transactions_data)
        
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Make predictions
        predictions_df = pred.predict(
            df,
            model_name=request.model_name,
            include_recommendations=request.include_recommendations
        )
        
        # Convert to response format
        predictions = []
        for _, row in predictions_df.iterrows():
            loan_rec = None
            if request.include_recommendations and 'recommended_loan_amount' in row:
                loan_rec = LoanRecommendation(
                    recommended_loan_amount=float(row['recommended_loan_amount']),
                    recommended_loan_duration_days=int(row['recommended_loan_duration_days'])
                )
            
            predictions.append(
                PredictionResult(
                    CustomerId=str(row['CustomerId']),
                    risk_probability=float(row['risk_probability']),
                    credit_score=int(row['credit_score']),
                    risk_category=str(row['risk_category']),
                    loan_recommendation=loan_rec
                )
            )
        
        return PredictionResponse(
            predictions=predictions,
            model_used=request.model_name,
            total_customers=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(request: PredictionRequest):
    """
    Batch prediction endpoint (same as /predict but with different response format).
    """
    return await predict(request)


@app.get("/models")
async def list_models():
    """List available models."""
    pred = get_predictor()
    model_dir = pred.model_dir
    
    available_models = []
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            if model_file.stem not in ['scaler', 'processor']:
                available_models.append(model_file.stem)
    
    return {
        "available_models": available_models,
        "loaded_models": list(pred.models.keys()) if pred.models else []
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
