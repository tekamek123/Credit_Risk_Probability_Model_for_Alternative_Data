"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import CreditRiskDataProcessor


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    dates = pd.date_range(start='2018-11-15', periods=100, freq='D')
    
    data = {
        'TransactionId': [f'TransactionId_{i}' for i in range(100)],
        'BatchId': [f'BatchId_{i}' for i in range(100)],
        'AccountId': [f'AccountId_{i%10}' for i in range(100)],
        'SubscriptionId': [f'SubscriptionId_{i%10}' for i in range(100)],
        'CustomerId': [f'CustomerId_{i%5}' for i in range(100)],  # 5 customers
        'CurrencyCode': ['UGX'] * 100,
        'CountryCode': [256] * 100,
        'ProviderId': [f'ProviderId_{i%6}' for i in range(100)],
        'ProductId': [f'ProductId_{i%10}' for i in range(100)],
        'ProductCategory': ['airtime', 'utility_bill', 'financial_services'] * 33 + ['airtime'],
        'ChannelId': [f'ChannelId_{i%3}' for i in range(100)],
        'Amount': np.random.normal(1000, 500, 100),
        'Value': np.abs(np.random.normal(1000, 500, 100)).astype(int),
        'TransactionStartTime': dates,
        'PricingStrategy': [2] * 100,
        'FraudResult': [0] * 95 + [1] * 5  # 5% fraud
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def processor():
    """Create a CreditRiskDataProcessor instance."""
    return CreditRiskDataProcessor()


class TestDataLoading:
    """Tests for data loading functionality."""
    
    def test_load_data(self, processor, tmp_path):
        """Test loading data from CSV."""
        # Create temporary CSV file
        sample_data = pd.DataFrame({
            'TransactionId': ['T1', 'T2'],
            'BatchId': ['B1', 'B2'],
            'AccountId': ['A1', 'A2'],
            'SubscriptionId': ['S1', 'S2'],
            'CustomerId': ['C1', 'C2'],
            'CurrencyCode': ['UGX', 'UGX'],
            'CountryCode': [256, 256],
            'ProviderId': ['P1', 'P2'],
            'ProductId': ['Prod1', 'Prod2'],
            'ProductCategory': ['airtime', 'utility'],
            'ChannelId': ['Ch1', 'Ch2'],
            'Amount': [1000.0, -500.0],
            'Value': [1000, 500],
            'TransactionStartTime': ['2018-11-15T02:18:49Z', '2018-11-16T02:18:49Z'],
            'PricingStrategy': [2, 2],
            'FraudResult': [0, 0]
        })
        
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Load data
        df = processor.load_data(str(csv_path))
        
        assert len(df) == 2
        assert 'TransactionStartTime' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime'])


class TestProxyVariable:
    """Tests for proxy variable creation."""
    
    def test_create_proxy_variable(self, processor, sample_transaction_data):
        """Test proxy variable creation."""
        df = processor.create_proxy_variable(sample_transaction_data)
        
        assert 'is_high_risk' in df.columns
        assert df['is_high_risk'].dtype in [int, np.int64]
        assert df['is_high_risk'].isin([0, 1]).all()
    
    def test_proxy_variable_with_fraud(self, processor, sample_transaction_data):
        """Test that customers with fraud are marked as high risk."""
        # Add fraud to one customer
        sample_transaction_data.loc[0, 'FraudResult'] = 1
        
        df = processor.create_proxy_variable(sample_transaction_data)
        
        # Customer with fraud should be high risk
        fraud_customer = df[df['FraudResult'] == 1]['CustomerId'].iloc[0]
        customer_risk = df[df['CustomerId'] == fraud_customer]['is_high_risk'].iloc[0]
        
        assert customer_risk == 1


class TestFeatureEngineering:
    """Tests for feature engineering."""
    
    def test_engineer_features(self, processor, sample_transaction_data):
        """Test feature engineering."""
        df = processor.engineer_features(sample_transaction_data, is_training=True)
        
        # Check RFM features
        assert 'recency_days' in df.columns
        assert 'frequency' in df.columns
        assert 'monetary' in df.columns
        
        # Check temporal features
        assert 'year' in df.columns
        assert 'month' in df.columns
        assert 'day_of_week' in df.columns
        assert 'hour' in df.columns
        assert 'is_weekend' in df.columns
        
        # Check transaction features
        assert 'is_debit' in df.columns
        assert 'is_credit' in df.columns
        assert 'amount_log' in df.columns
    
    def test_categorical_encoding(self, processor, sample_transaction_data):
        """Test categorical encoding."""
        df = processor.engineer_features(sample_transaction_data, is_training=True)
        
        # Check that encoders are created
        assert 'ProductCategory' in processor.categorical_encoders
        assert 'ChannelId' in processor.categorical_encoders
        
        # Check encoded features exist
        assert 'product_category_fraud_rate' in df.columns
        assert 'channel_fraud_rate' in df.columns


class TestFeatureSelection:
    """Tests for feature selection."""
    
    def test_select_features(self, processor, sample_transaction_data):
        """Test feature selection."""
        # First engineer features
        df = processor.engineer_features(sample_transaction_data, is_training=True)
        
        # Then select features
        selected_df = processor.select_features(df)
        
        # Check that feature columns are set
        assert processor.feature_columns is not None
        assert len(processor.feature_columns) > 0
        
        # Check that selected DataFrame has correct columns
        assert 'CustomerId' in selected_df.columns
        assert 'is_high_risk' in selected_df.columns
        assert all(col in selected_df.columns for col in processor.feature_columns)
    
    def test_feature_selection_no_nan(self, processor, sample_transaction_data):
        """Test that selected features have no NaN values."""
        df = processor.engineer_features(sample_transaction_data, is_training=True)
        selected_df = processor.select_features(df)
        
        # Check for NaN in feature columns
        feature_cols = [col for col in processor.feature_columns if col in selected_df.columns]
        assert selected_df[feature_cols].isnull().sum().sum() == 0


class TestDataProcessingPipeline:
    """Tests for complete data processing pipeline."""
    
    def test_process_data_training(self, processor, tmp_path):
        """Test complete processing pipeline for training."""
        # Create test data file
        sample_data = pd.DataFrame({
            'TransactionId': ['T1', 'T2', 'T3'],
            'BatchId': ['B1', 'B2', 'B3'],
            'AccountId': ['A1', 'A2', 'A3'],
            'SubscriptionId': ['S1', 'S2', 'S3'],
            'CustomerId': ['C1', 'C1', 'C2'],
            'CurrencyCode': ['UGX'] * 3,
            'CountryCode': [256] * 3,
            'ProviderId': ['P1', 'P2', 'P1'],
            'ProductId': ['Prod1', 'Prod2', 'Prod1'],
            'ProductCategory': ['airtime', 'utility', 'airtime'],
            'ChannelId': ['Ch1', 'Ch2', 'Ch1'],
            'Amount': [1000.0, -500.0, 2000.0],
            'Value': [1000, 500, 2000],
            'TransactionStartTime': [
                '2018-11-15T02:18:49Z',
                '2018-11-16T02:18:49Z',
                '2018-11-17T02:18:49Z'
            ],
            'PricingStrategy': [2, 2, 2],
            'FraudResult': [0, 0, 1]
        })
        
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Process data
        X, y = processor.process_data(str(csv_path), is_training=True)
        
        # Check outputs
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(X.columns) > 0
        assert y.isin([0, 1]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
