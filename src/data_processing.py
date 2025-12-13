"""
Data processing and feature engineering module for credit risk modeling.
Implements RFM-based feature engineering and proxy variable creation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CreditRiskDataProcessor:
    """
    Data processor for credit risk modeling.
    Handles feature engineering, RFM analysis, and proxy variable creation.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.feature_columns = None
        self.categorical_encoders = {}
        self.scaler = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load transaction data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with transaction data
        """
        df = pd.read_csv(file_path)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        return df
    
    def create_proxy_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proxy variable for credit risk (high risk = 1, low risk = 0).
        
        Based on RFM patterns and fraud indicators:
        - High risk: Has fraud history OR low engagement OR irregular patterns
        - Low risk: No fraud, consistent engagement, stable patterns
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            DataFrame with proxy variable 'is_high_risk'
        """
        # Calculate customer-level features
        customer_features = self._calculate_customer_features(df)
        
        # Define high risk criteria
        # 1. Has fraud history
        has_fraud = customer_features['has_fraud'] == 1
        
        # 2. Low recency (inactive for >30 days) AND low frequency (<5 transactions)
        low_engagement = (customer_features['recency_days'] > 30) & (customer_features['transaction_count'] < 5)
        
        # 3. High transaction volatility (std > mean) AND low average amount
        high_volatility = (customer_features['amount_std'] > customer_features['amount_mean']) & (customer_features['amount_mean'] < 1000)
        
        # 4. Declining transaction pattern (recent transactions < 50% of total)
        declining_pattern = customer_features['recent_transaction_ratio'] < 0.5
        
        # Combine criteria: high risk if any condition is met
        customer_features['is_high_risk'] = (
            has_fraud | low_engagement | high_volatility | declining_pattern
        ).astype(int)
        
        # Merge back to transaction level
        df = df.merge(
            customer_features[['CustomerId', 'is_high_risk']],
            on='CustomerId',
            how='left'
        )
        
        # Fill any missing values (shouldn't happen, but safety check)
        df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)
        
        return df
    
    def _calculate_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate customer-level features for RFM analysis.
        
        Args:
            df: Transaction DataFrame
            
        Returns:
            DataFrame with customer-level features
        """
        # Reference date (most recent transaction date)
        reference_date = df['TransactionStartTime'].max()
        
        customer_features = df.groupby('CustomerId').agg({
            'TransactionId': 'count',
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'Value': ['sum', 'mean'],
            'FraudResult': 'sum',
            'TransactionStartTime': ['min', 'max'],
            'ProductCategory': lambda x: x.nunique(),
            'ChannelId': lambda x: x.nunique(),
            'ProviderId': lambda x: x.nunique()
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            'CustomerId', 'transaction_count', 'amount_sum', 'amount_mean', 
            'amount_std', 'amount_min', 'amount_max', 'value_sum', 'value_mean',
            'fraud_count', 'first_transaction', 'last_transaction',
            'unique_categories', 'unique_channels', 'unique_providers'
        ]
        
        # Calculate RFM features
        customer_features['recency_days'] = (
            reference_date - customer_features['last_transaction']
        ).dt.days
        
        customer_features['frequency'] = customer_features['transaction_count']
        
        customer_features['monetary'] = customer_features['amount_sum']
        
        # Additional features
        customer_features['has_fraud'] = (customer_features['fraud_count'] > 0).astype(int)
        
        customer_features['avg_transaction_value'] = customer_features['value_mean']
        
        customer_features['transaction_span_days'] = (
            customer_features['last_transaction'] - customer_features['first_transaction']
        ).dt.days
        
        customer_features['avg_days_between_transactions'] = (
            customer_features['transaction_span_days'] / 
            (customer_features['transaction_count'] - 1).replace(0, 1)
        )
        
        # Recent transaction ratio (last 30 days)
        recent_date = reference_date - pd.Timedelta(days=30)
        recent_transactions = df[df['TransactionStartTime'] >= recent_date].groupby('CustomerId').size()
        customer_features = customer_features.merge(
            recent_transactions.to_frame('recent_transaction_count'),
            left_on='CustomerId',
            right_index=True,
            how='left'
        )
        customer_features['recent_transaction_count'] = customer_features['recent_transaction_count'].fillna(0)
        customer_features['recent_transaction_ratio'] = (
            customer_features['recent_transaction_count'] / 
            customer_features['transaction_count'].replace(0, 1)
        )
        
        # Fill NaN values
        customer_features['amount_std'] = customer_features['amount_std'].fillna(0)
        customer_features['avg_days_between_transactions'] = customer_features['avg_days_between_transactions'].fillna(0)
        
        return customer_features
    
    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Engineer features for model training/inference.
        
        Args:
            df: Transaction DataFrame
            is_training: Whether this is training data (affects encoding fitting)
            
        Returns:
            DataFrame with engineered features
        """
        # Create proxy variable if not exists
        if 'is_high_risk' not in df.columns:
            df = self.create_proxy_variable(df)
        
        # Calculate customer-level features
        customer_features = self._calculate_customer_features(df)
        
        # Merge customer features to transaction level
        df = df.merge(customer_features, on='CustomerId', how='left')
        
        # Temporal features
        df['year'] = df['TransactionStartTime'].dt.year
        df['month'] = df['TransactionStartTime'].dt.month
        df['day_of_week'] = df['TransactionStartTime'].dt.dayofweek
        df['hour'] = df['TransactionStartTime'].dt.hour
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Transaction features
        df['is_debit'] = (df['Amount'] > 0).astype(int)
        df['is_credit'] = (df['Amount'] < 0).astype(int)
        df['amount_abs'] = df['Amount'].abs()
        df['amount_log'] = np.log1p(df['amount_abs'])
        
        # Amount categories
        df['amount_category'] = pd.cut(
            df['amount_abs'],
            bins=[0, 100, 1000, 10000, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Categorical encoding (target encoding for ProductCategory and ChannelId)
        if is_training:
            # Calculate fraud rates for target encoding
            product_fraud_rate = df.groupby('ProductCategory')['FraudResult'].mean()
            channel_fraud_rate = df.groupby('ChannelId')['FraudResult'].mean()
            
            self.categorical_encoders['ProductCategory'] = product_fraud_rate
            self.categorical_encoders['ChannelId'] = channel_fraud_rate
        else:
            # Use pre-fitted encoders
            product_fraud_rate = self.categorical_encoders.get('ProductCategory', pd.Series())
            channel_fraud_rate = self.categorical_encoders.get('ChannelId', pd.Series())
        
        df['product_category_fraud_rate'] = df['ProductCategory'].map(product_fraud_rate).fillna(0)
        df['channel_fraud_rate'] = df['ChannelId'].map(channel_fraud_rate).fillna(0)
        
        # Interaction features
        df['amount_category_encoded'] = df['amount_category'].cat.codes
        df['amount_channel_interaction'] = df['amount_log'] * df['channel_fraud_rate']
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select final features for model training.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            DataFrame with selected features
        """
        # Define feature columns
        feature_columns = [
            # RFM features
            'recency_days', 'frequency', 'monetary',
            'transaction_count', 'avg_transaction_value',
            'transaction_span_days', 'avg_days_between_transactions',
            
            # Transaction features
            'amount_log', 'amount_abs', 'is_debit', 'is_credit',
            'amount_category_encoded',
            
            # Behavioral features
            'unique_categories', 'unique_channels', 'unique_providers',
            'recent_transaction_ratio',
            
            # Risk indicators
            'has_fraud', 'product_category_fraud_rate', 'channel_fraud_rate',
            'amount_channel_interaction',
            
            # Temporal features
            'month', 'day_of_week', 'hour', 'is_weekend',
            
            # Statistical features
            'amount_mean', 'amount_std', 'amount_min', 'amount_max',
        ]
        
        # Ensure all columns exist
        available_features = [col for col in feature_columns if col in df.columns]
        
        if self.feature_columns is None:
            self.feature_columns = available_features
        
        # Select features
        selected_df = df[['CustomerId', 'is_high_risk'] + self.feature_columns].copy()
        
        # Fill any remaining NaN values
        selected_df = selected_df.fillna(0)
        
        return selected_df
    
    def process_data(
        self, 
        file_path: str, 
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete data processing pipeline.
        
        Args:
            file_path: Path to CSV file
            is_training: Whether this is training data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Load data
        df = self.load_data(file_path)
        
        # Engineer features
        df = self.engineer_features(df, is_training=is_training)
        
        # Select features
        df = self.select_features(df)
        
        # Separate features and target
        if is_training:
            X = df.drop(['CustomerId', 'is_high_risk'], axis=1, errors='ignore')
            y = df['is_high_risk']
        else:
            X = df.drop(['CustomerId'], axis=1, errors='ignore')
            y = None
        
        return X, y
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self.feature_columns if self.feature_columns else []
