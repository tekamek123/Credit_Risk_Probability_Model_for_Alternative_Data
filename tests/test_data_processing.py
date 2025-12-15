"""
Unit tests for data processing module with sklearn Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import (
    CreditRiskDataProcessor,
    AggregateFeatureTransformer,
    TemporalFeatureExtractor,
    CategoricalEncoder,
    MissingValueHandler,
    WOETransformer
)


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
    return CreditRiskDataProcessor(use_woe=False, encoding_type='onehot')


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
    """Tests for proxy variable creation using K-Means clustering."""
    
    def test_create_proxy_variable(self, processor, sample_transaction_data):
        """Test proxy variable creation using K-Means clustering."""
        df = processor.create_proxy_variable(sample_transaction_data, n_clusters=3, random_state=42)
        
        # Check that is_high_risk column exists
        assert 'is_high_risk' in df.columns
        assert df['is_high_risk'].dtype in [int, np.int64]
        assert df['is_high_risk'].isin([0, 1]).all()
        
        # Check that RFM metrics are present
        assert 'Recency' in df.columns
        assert 'Frequency' in df.columns
        assert 'Monetary' in df.columns
        assert 'cluster' in df.columns
        
        # Check that cluster info is stored
        cluster_info = processor.get_cluster_info()
        assert cluster_info is not None
        assert 'high_risk_cluster' in cluster_info
        assert 'cluster_summary' in cluster_info
        assert 'cluster_centers' in cluster_info
    
    def test_rfm_metrics_calculation(self, processor, sample_transaction_data):
        """Test that RFM metrics are calculated correctly."""
        df = processor.create_proxy_variable(sample_transaction_data, n_clusters=3, random_state=42)
        
        # Check RFM metrics are positive
        assert (df['Recency'] >= 0).all()
        assert (df['Frequency'] > 0).all()
        assert (df['Monetary'] > 0).all()
        
        # Check that each customer has RFM values
        assert df['Recency'].notna().all()
        assert df['Frequency'].notna().all()
        assert df['Monetary'].notna().all()
    
    def test_kmeans_clustering(self, processor, sample_transaction_data):
        """Test that K-Means clustering creates 3 distinct clusters."""
        df = processor.create_proxy_variable(sample_transaction_data, n_clusters=3, random_state=42)
        
        # Check that cluster column exists and has 3 unique values
        unique_clusters = df['cluster'].unique()
        assert len(unique_clusters) == 3
        assert set(unique_clusters).issubset({0, 1, 2})
        
        # Check that high-risk cluster is identified
        cluster_info = processor.get_cluster_info()
        high_risk_cluster = cluster_info['high_risk_cluster']
        assert high_risk_cluster in [0, 1, 2]
        
        # Check that high-risk customers are in the identified cluster
        high_risk_customers = df[df['is_high_risk'] == 1]
        if len(high_risk_customers) > 0:
            assert (high_risk_customers['cluster'] == high_risk_cluster).all()
    
    def test_proxy_variable_reproducibility(self, processor, sample_transaction_data):
        """Test that proxy variable creation is reproducible with same random_state."""
        df1 = processor.create_proxy_variable(sample_transaction_data, n_clusters=3, random_state=42)
        processor2 = CreditRiskDataProcessor()
        df2 = processor2.create_proxy_variable(sample_transaction_data, n_clusters=3, random_state=42)
        
        # Check that high-risk assignments are the same
        assert (df1['is_high_risk'] == df2['is_high_risk']).all()
        assert (df1['cluster'] == df2['cluster']).all()


class TestAggregateFeatures:
    """Tests for aggregate feature transformer."""
    
    def test_aggregate_transformer(self, sample_transaction_data):
        """Test AggregateFeatureTransformer."""
        transformer = AggregateFeatureTransformer()
        transformer.fit(sample_transaction_data)
        df_transformed = transformer.transform(sample_transaction_data)
        
        # Check required aggregate features
        assert 'total_amount' in df_transformed.columns
        assert 'avg_amount' in df_transformed.columns
        assert 'transaction_count' in df_transformed.columns
        assert 'amount_std' in df_transformed.columns
        
        # Check that values are calculated correctly
        assert df_transformed['total_amount'].notna().all()
        assert df_transformed['transaction_count'].notna().all()


class TestTemporalFeatures:
    """Tests for temporal feature extractor."""
    
    def test_temporal_extractor(self, sample_transaction_data):
        """Test TemporalFeatureExtractor."""
        transformer = TemporalFeatureExtractor()
        transformer.fit(sample_transaction_data)
        df_transformed = transformer.transform(sample_transaction_data)
        
        # Check required temporal features
        assert 'transaction_hour' in df_transformed.columns
        assert 'transaction_day' in df_transformed.columns
        assert 'transaction_month' in df_transformed.columns
        assert 'transaction_year' in df_transformed.columns
        
        # Check value ranges
        assert df_transformed['transaction_hour'].between(0, 23).all()
        assert df_transformed['transaction_day'].between(1, 31).all()
        assert df_transformed['transaction_month'].between(1, 12).all()
        assert df_transformed['transaction_year'].min() >= 2018


class TestCategoricalEncoding:
    """Tests for categorical encoding."""
    
    def test_onehot_encoding(self, sample_transaction_data):
        """Test One-Hot encoding."""
        encoder = CategoricalEncoder(
            columns=['ProductCategory', 'ChannelId'],
            encoding_type='onehot'
        )
        encoder.fit(sample_transaction_data)
        df_encoded = df_encoded = encoder.transform(sample_transaction_data)
        
        # Check that original columns are removed
        assert 'ProductCategory' not in df_encoded.columns
        assert 'ChannelId' not in df_encoded.columns
        
        # Check that encoded columns exist
        encoded_cols = [col for col in df_encoded.columns 
                       if 'ProductCategory' in col or 'ChannelId' in col]
        assert len(encoded_cols) > 0
    
    def test_label_encoding(self, sample_transaction_data):
        """Test Label encoding."""
        encoder = CategoricalEncoder(
            columns=['ProductCategory'],
            encoding_type='label'
        )
        encoder.fit(sample_transaction_data)
        df_encoded = encoder.transform(sample_transaction_data)
        
        # Check that encoded column exists
        assert 'ProductCategory_encoded' in df_encoded.columns
        assert df_encoded['ProductCategory_encoded'].dtype in [int, np.int64]


class TestMissingValueHandling:
    """Tests for missing value handling."""
    
    def test_missing_value_imputation(self, sample_transaction_data):
        """Test missing value imputation."""
        # Add some missing values
        sample_transaction_data.loc[0:5, 'Amount'] = np.nan
        
        handler = MissingValueHandler(strategy='mean')
        handler.fit(sample_transaction_data)
        df_imputed = handler.transform(sample_transaction_data)
        
        # Check that missing values are filled
        assert df_imputed['Amount'].notna().all()
    
    def test_missing_value_removal(self, sample_transaction_data):
        """Test missing value removal."""
        # Add many missing values to one column
        sample_transaction_data.loc[0:60, 'Amount'] = np.nan
        
        handler = MissingValueHandler(strategy='remove', columns=['Amount'])
        handler.fit(sample_transaction_data)
        df_cleaned = handler.transform(sample_transaction_data)
        
        # Check that missing values are handled
        # If column was dropped (>50% missing), it won't be in df_cleaned
        # If column was kept, missing values should be removed via row removal
        if 'Amount' in df_cleaned.columns:
            assert df_cleaned['Amount'].notna().all()
        else:
            # Column was dropped due to high missing percentage
            assert True


class TestPipeline:
    """Tests for complete sklearn Pipeline."""
    
    def test_pipeline_build(self, processor):
        """Test pipeline building."""
        pipeline = processor.build_pipeline()
        
        assert pipeline is not None
        assert len(pipeline.steps) > 0
        assert 'temporal' in [step[0] for step in pipeline.steps]
        assert 'aggregate' in [step[0] for step in pipeline.steps]
        assert 'missing_values' in [step[0] for step in pipeline.steps]
        assert 'categorical' in [step[0] for step in pipeline.steps] or 'woe' in [step[0] for step in pipeline.steps]
    
    def test_process_data_training(self, processor, tmp_path):
        """Test complete processing pipeline for training."""
        # Create test data file
        sample_data = pd.DataFrame({
            'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'BatchId': ['B1', 'B2', 'B3', 'B4', 'B5'],
            'AccountId': ['A1', 'A2', 'A3', 'A4', 'A5'],
            'SubscriptionId': ['S1', 'S2', 'S3', 'S4', 'S5'],
            'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
            'CurrencyCode': ['UGX'] * 5,
            'CountryCode': [256] * 5,
            'ProviderId': ['P1', 'P2', 'P1', 'P2', 'P1'],
            'ProductId': ['Prod1', 'Prod2', 'Prod1', 'Prod2', 'Prod1'],
            'ProductCategory': ['airtime', 'utility', 'airtime', 'utility', 'airtime'],
            'ChannelId': ['Ch1', 'Ch2', 'Ch1', 'Ch2', 'Ch1'],
            'Amount': [1000.0, -500.0, 2000.0, 1500.0, 3000.0],
            'Value': [1000, 500, 2000, 1500, 3000],
            'TransactionStartTime': [
                '2018-11-15T02:18:49Z',
                '2018-11-16T02:18:49Z',
                '2018-11-17T02:18:49Z',
                '2018-11-18T02:18:49Z',
                '2018-11-19T02:18:49Z'
            ],
            'PricingStrategy': [2, 2, 2, 2, 2],
            'FraudResult': [0, 0, 1, 0, 0]
        })
        
        csv_path = tmp_path / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        # Process data
        X, y = processor.process_data(str(csv_path), is_training=True, scaling_method='standard')
        
        # Check outputs
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(X.columns) > 0
        assert y.isin([0, 1]).all()
        
        # Check that aggregate features are present
        assert any('total_amount' in str(col) or 'avg_amount' in str(col) or 
                  'transaction_count' in str(col) for col in X.columns)
        
        # Check that temporal features are present
        assert any('hour' in str(col) or 'day' in str(col) or 
                  'month' in str(col) or 'year' in str(col) for col in X.columns)
        
        # Check that scaling is applied
        assert processor.scaler is not None
    
    def test_process_data_inference(self, processor, tmp_path):
        """Test processing pipeline for inference (no target)."""
        # Create test data file
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
        
        # First fit on training data
        processor.process_data(str(csv_path), is_training=True)
        
        # Then transform for inference
        X, y = processor.process_data(str(csv_path), is_training=False)
        
        # Check outputs
        assert X is not None
        assert y is None  # No target for inference
        assert len(X.columns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
