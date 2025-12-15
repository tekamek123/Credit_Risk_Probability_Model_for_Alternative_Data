"""
Prediction script for credit risk model inference.
Provides functions for risk probability, credit score, and loan recommendations.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.data_processing import CreditRiskDataProcessor


class CreditRiskPredictor:
    """
    Predictor for credit risk models.
    Handles risk probability, credit score, and loan recommendations.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.processor = None
        self.feature_names = None
        self.model_name = 'logistic_regression'  # Default model
        self.mlflow_model = None  # MLflow model
        self.model_source = None  # 'mlflow' or 'local'
        
    def load_models(self, model_name: str = 'logistic_regression'):
        """
        Load trained models and preprocessing objects.
        
        Args:
            model_name: Name of model to load ('logistic_regression', 'xgboost', 'lightgbm')
        """
        self.model_name = model_name
        
        # Load model
        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        
        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load processor
        processor_path = self.model_dir / "processor.pkl"
        if processor_path.exists():
            with open(processor_path, 'rb') as f:
                self.processor = pickle.load(f)
        
        # Load feature names
        feature_names_path = self.model_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        
        print(f"Loaded {model_name} model successfully")
    
    def predict_risk_probability(
        self, 
        data: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict risk probability for given data.
        
        Args:
            data: DataFrame with transaction data
            model_name: Model to use (default: self.model_name)
            
        Returns:
            Array of risk probabilities (0-1)
        """
        # Use MLflow model if available
        if self.mlflow_model is not None:
            # Process data
            if self.processor is None:
                # Initialize processor if not loaded
                self.processor = CreditRiskDataProcessor()
            X, _ = self.processor.process_data_from_df(data, is_training=False)
            
            # MLflow models expect DataFrame input
            try:
                # Try to get probabilities
                if hasattr(self.mlflow_model, 'predict_proba'):
                    probabilities = self.mlflow_model.predict_proba(X)[:, 1]
                elif hasattr(self.mlflow_model, 'predict'):
                    # Get predictions and convert to probabilities
                    predictions = self.mlflow_model.predict(X)
                    probabilities = predictions.astype(float)
                else:
                    # Use pyfunc interface
                    probabilities = self.mlflow_model.predict(X)
                    if probabilities.ndim > 1:
                        probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            except Exception as e:
                # Fallback: try direct prediction
                probabilities = self.mlflow_model.predict(X)
                if isinstance(probabilities, pd.DataFrame):
                    probabilities = probabilities.values.flatten()
            
            return np.array(probabilities)
        
        # Fallback to local model
        if model_name is None:
            model_name = self.model_name
        
        if model_name not in self.models:
            self.load_models(model_name)
        
        # Process data
        if self.processor is None:
            raise ValueError("Processor not loaded. Cannot process data.")
        X, _ = self.processor.process_data_from_df(data, is_training=False)
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        model = self.models[model_name]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = model.predict(X_scaled)
        
        return probabilities
    
    def predict_credit_score(
        self, 
        risk_probability: np.ndarray,
        min_score: int = 300,
        max_score: int = 850
    ) -> np.ndarray:
        """
        Convert risk probability to credit score.
        
        Lower risk probability = Higher credit score
        
        Args:
            risk_probability: Array of risk probabilities (0-1)
            min_score: Minimum credit score (default: 300)
            max_score: Maximum credit score (default: 850)
            
        Returns:
            Array of credit scores
        """
        # Invert probability: low risk (low prob) = high score
        # Scale to score range
        scores = min_score + (1 - risk_probability) * (max_score - min_score)
        return scores.astype(int)
    
    def recommend_loan(
        self,
        risk_probability: np.ndarray,
        customer_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Recommend optimal loan amount and duration based on risk.
        
        Args:
            risk_probability: Array of risk probabilities
            customer_features: Optional DataFrame with customer features
            
        Returns:
            Dictionary with 'loan_amount' and 'loan_duration_days' arrays
        """
        # Base loan amount on risk and customer monetary value
        base_amount = 10000  # Base loan amount
        
        # Adjust based on risk: lower risk = higher loan amount
        risk_adjustment = 1 - risk_probability
        loan_amounts = base_amount * (0.5 + 1.5 * risk_adjustment)  # Range: 0.5x to 2x base
        
        # If customer features available, adjust based on average transaction value
        if customer_features is not None and 'monetary' in customer_features.columns:
            avg_monetary = customer_features['monetary'].values
            # Cap loan amount at 3x average monthly spending
            monthly_spending = avg_monetary / 3  # Approximate monthly from total
            loan_amounts = np.minimum(loan_amounts, monthly_spending * 3)
        
        # Round to nearest 1000
        loan_amounts = np.round(loan_amounts / 1000) * 1000
        loan_amounts = np.maximum(loan_amounts, 1000)  # Minimum 1000
        loan_amounts = np.minimum(loan_amounts, 100000)  # Maximum 100000
        
        # Loan duration: lower risk = longer duration
        # Range: 30 days (high risk) to 365 days (low risk)
        loan_durations = 30 + (1 - risk_probability) * 335
        loan_durations = np.round(loan_durations / 30) * 30  # Round to nearest month
        loan_durations = np.maximum(loan_durations, 30)  # Minimum 30 days
        loan_durations = np.minimum(loan_durations, 365)  # Maximum 365 days
        
        return {
            'loan_amount': loan_amounts.astype(int),
            'loan_duration_days': loan_durations.astype(int)
        }
    
    def predict(
        self,
        data: pd.DataFrame,
        model_name: Optional[str] = None,
        include_recommendations: bool = True
    ) -> pd.DataFrame:
        """
        Complete prediction pipeline.
        
        Args:
            data: DataFrame with transaction data
            model_name: Model to use
            include_recommendations: Whether to include loan recommendations
            
        Returns:
            DataFrame with predictions
        """
        # Predict risk probability
        risk_probability = self.predict_risk_probability(data, model_name)
        
        # Convert to credit score
        credit_score = self.predict_credit_score(risk_probability)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'CustomerId': data['CustomerId'].values if 'CustomerId' in data.columns else range(len(data)),
            'risk_probability': risk_probability,
            'credit_score': credit_score,
            'risk_category': pd.cut(
                risk_probability,
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
        })
        
        # Add loan recommendations if requested
        if include_recommendations:
            # Get customer features for loan recommendation
            customer_features = None
            if self.processor is not None:
                try:
                    processed_data = self.processor.engineer_features(data, is_training=False)
                    customer_features = processed_data.groupby('CustomerId').agg({
                        'monetary': 'first'
                    }).reset_index()
                except:
                    pass
            
            loan_recs = self.recommend_loan(risk_probability, customer_features)
            results['recommended_loan_amount'] = loan_recs['loan_amount']
            results['recommended_loan_duration_days'] = loan_recs['loan_duration_days']
        
        return results


# Add helper method to processor for inference
def process_data_from_df(self, df: pd.DataFrame, is_training: bool = False):
    """Process data from DataFrame (for inference)."""
    # Engineer features
    df = self.engineer_features(df, is_training=is_training)
    
    # Select features
    df = self.select_features(df)
    
    # Separate features
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1, errors='ignore')
    
    return X, None

# Add method to processor class
CreditRiskDataProcessor.process_data_from_df = process_data_from_df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data_path> [model_name]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'logistic_regression'
    
    # Load data
    processor = CreditRiskDataProcessor()
    data = processor.load_data(data_path)
    
    # Predict
    predictor = CreditRiskPredictor()
    predictor.load_models(model_name)
    
    # Sample prediction (use first customer for demo)
    sample_customer = data[data['CustomerId'] == data['CustomerId'].iloc[0]]
    predictions = predictor.predict(sample_customer)
    
    print("\nPredictions:")
    print(predictions.to_string())
