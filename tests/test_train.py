"""
Unit tests for model training module.
Tests helper functions and training pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.train import CreditRiskModelTrainer
from src.data_processing import CreditRiskDataProcessor
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_data(tmp_path):
    """Create sample data file for testing."""
    # Create sample transaction data
    dates = pd.date_range(start='2018-11-15', periods=200, freq='D')
    
    data = {
        'TransactionId': [f'T{i}' for i in range(200)],
        'BatchId': [f'B{i}' for i in range(200)],
        'AccountId': [f'A{i%20}' for i in range(200)],
        'SubscriptionId': [f'S{i%20}' for i in range(200)],
        'CustomerId': [f'C{i%10}' for i in range(200)],  # 10 customers
        'CurrencyCode': ['UGX'] * 200,
        'CountryCode': [256] * 200,
        'ProviderId': [f'P{i%5}' for i in range(200)],
        'ProductId': [f'Prod{i%10}' for i in range(200)],
        'ProductCategory': ['airtime', 'utility', 'financial'] * 66 + ['airtime', 'utility'],
        'ChannelId': [f'Ch{i%3}' for i in range(200)],
        'Amount': np.random.normal(1000, 500, 200),
        'Value': np.abs(np.random.normal(1000, 500, 200)).astype(int),
        'TransactionStartTime': dates,
        'PricingStrategy': [2] * 200,
        'FraudResult': [0] * 180 + [1] * 20  # 10% fraud
    }
    
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def trainer():
    """Create a CreditRiskModelTrainer instance."""
    return CreditRiskModelTrainer(model_dir="test_models")


class TestHelperFunctions:
    """Tests for helper functions in training module."""
    
    def test_calculate_metrics(self, trainer, sample_data):
        """Test that _calculate_metrics returns expected metrics."""
        # Process data and split
        processor = CreditRiskDataProcessor()
        X, y = processor.process_data(sample_data, is_training=True)
        
        # Ensure X only contains numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Check we have enough samples
        if len(X) < 10:
            pytest.skip("Not enough samples for testing")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = trainer._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        assert all(metric in metrics for metric in expected_metrics)
        
        # Check that metrics are in valid ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_calculate_metrics_with_perfect_predictions(self, trainer):
        """Test _calculate_metrics with perfect predictions."""
        # Create perfect predictions
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.9])
        
        metrics = trainer._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # With perfect predictions, accuracy should be 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_identify_best_model(self, trainer):
        """Test that _identify_best_model correctly identifies best model."""
        # Create mock results
        trainer.models = {
            'model1': 'mock_model_1',
            'model2': 'mock_model_2',
            'model3': 'mock_model_3'
        }
        
        results = {
            'model1': {'roc_auc': 0.75, 'accuracy': 0.80},
            'model2': {'roc_auc': 0.85, 'accuracy': 0.82},  # Best
            'model3': {'roc_auc': 0.70, 'accuracy': 0.78}
        }
        
        trainer._identify_best_model(results)
        
        assert trainer.best_model_name == 'model2'
        assert trainer.best_score == 0.85
        assert trainer.best_model == 'mock_model_2'
    
    def test_data_splitting_reproducibility(self, trainer, sample_data):
        """Test that data splitting is reproducible with random_state."""
        processor = CreditRiskDataProcessor()
        X, y = processor.process_data(sample_data, is_training=True)
        
        # Split data twice with same random_state
        from sklearn.model_selection import train_test_split
        
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Check that splits are identical
        assert X1_train.equals(X2_train)
        assert X1_test.equals(X2_test)
        assert y1_train.equals(y2_train)
        assert y1_test.equals(y2_test)


class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    def test_feature_engineering_returns_expected_columns(self, sample_data):
        """Test that feature engineering returns expected columns."""
        processor = CreditRiskDataProcessor()
        X, y = processor.process_data(sample_data, is_training=True)
        
        # Check that X is a DataFrame
        assert isinstance(X, pd.DataFrame)
        
        # Check that X has features (not empty)
        assert len(X.columns) > 0
        
        # Check that X doesn't contain ID columns
        id_columns = ['CustomerId', 'AccountId', 'SubscriptionId', 'TransactionId', 'BatchId']
        assert not any(col in X.columns for col in id_columns)
        
        # Check that X doesn't contain target column
        assert 'is_high_risk' not in X.columns
        
        # Check that y is a Series
        assert isinstance(y, pd.Series)
        
        # Check that y contains binary values
        assert set(y.unique()).issubset({0, 1})
    
    def test_feature_engineering_no_missing_values(self, sample_data):
        """Test that feature engineering produces no missing values."""
        processor = CreditRiskDataProcessor()
        X, y = processor.process_data(sample_data, is_training=True)
        
        # Check for missing values
        assert X.isnull().sum().sum() == 0, "Feature engineering produced missing values"
        assert y.isnull().sum() == 0, "Target variable has missing values"


class TestModelTraining:
    """Tests for model training pipeline."""
    
    def test_trainer_initialization(self):
        """Test that trainer initializes correctly."""
        trainer = CreditRiskModelTrainer(model_dir="test_models")
        
        assert trainer.model_dir.exists()
        assert trainer.processor is not None
        assert trainer.scaler is not None
        assert trainer.models == {}
        assert trainer.metrics == {}
    
    @pytest.mark.slow
    def test_train_multiple_models(self, trainer, sample_data):
        """Test training multiple models."""
        # Train with hyperparameter tuning disabled for speed
        results = trainer.train(
            sample_data,
            test_size=0.2,
            random_state=42,
            use_hyperparameter_tuning=False
        )
        
        # Check that results contain expected models
        expected_models = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost']
        for model_name in expected_models:
            assert model_name in results
        
        # Check that each model has metrics
        for model_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'roc_auc' in metrics
            
            # Check metric ranges
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['roc_auc'] <= 1


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_test_models():
    """Clean up test model directory after tests."""
    yield
    test_models_dir = Path("test_models")
    if test_models_dir.exists():
        shutil.rmtree(test_models_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

