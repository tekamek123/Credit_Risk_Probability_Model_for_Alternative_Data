"""
Data processing and feature engineering module for credit risk modeling.
Implements sklearn Pipeline for robust, automated, and reproducible transformations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

try:
    from xverse.transformer import WOE
    XVERSE_AVAILABLE = True
except ImportError:
    XVERSE_AVAILABLE = False
    WOE = None

try:
    from woe import WOE as WOE_ALT
    WOE_LIB_AVAILABLE = True
except ImportError:
    WOE_LIB_AVAILABLE = False
    WOE_ALT = None


class AggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create aggregate features at customer level.
    Creates: total_amount, avg_amount, transaction_count, amount_std
    """
    
    def __init__(self, customer_id_col: str = 'CustomerId', amount_col: str = 'Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.aggregate_features_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Calculate aggregate features from training data."""
        if isinstance(X, pd.DataFrame):
            grouped = X.groupby(self.customer_id_col)[self.amount_col]
            self.aggregate_features_ = pd.DataFrame({
                'total_amount': grouped.sum(),
                'avg_amount': grouped.mean(),
                'transaction_count': grouped.count(),
                'amount_std': grouped.std()
            }).fillna(0)  # Fill NaN std with 0 for single-transaction customers
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Merge aggregate features back to transaction level."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            if self.aggregate_features_ is not None:
                X = X.merge(
                    self.aggregate_features_,
                    left_on=self.customer_id_col,
                    right_index=True,
                    how='left'
                )
                # Fill any missing values (for new customers)
                agg_cols = ['total_amount', 'avg_amount', 'transaction_count', 'amount_std']
                for col in agg_cols:
                    if col in X.columns:
                        X[col] = X[col].fillna(0)
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract temporal features from TransactionStartTime.
    Extracts: hour, day, month, year
    """
    
    def __init__(self, datetime_col: str = 'TransactionStartTime'):
        self.datetime_col = datetime_col
        
    def fit(self, X: pd.DataFrame, y=None):
        """No fitting needed for temporal extraction."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            if self.datetime_col in X.columns:
                # Ensure datetime type
                if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_col]):
                    X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
                
                # Extract temporal features
                X['transaction_hour'] = X[self.datetime_col].dt.hour
                X['transaction_day'] = X[self.datetime_col].dt.day
                X['transaction_month'] = X[self.datetime_col].dt.month
                X['transaction_year'] = X[self.datetime_col].dt.year
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for categorical encoding.
    Supports One-Hot Encoding and Label Encoding.
    """
    
    def __init__(self, columns: List[str], encoding_type: str = 'onehot'):
        """
        Args:
            columns: List of categorical column names to encode
            encoding_type: 'onehot' or 'label'
        """
        self.columns = columns
        self.encoding_type = encoding_type
        self.encoders_ = {}
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit encoders on training data."""
        if isinstance(X, pd.DataFrame):
            for col in self.columns:
                if col in X.columns:
                    if self.encoding_type == 'onehot':
                        self.encoders_[col] = OneHotEncoder(
                            sparse_output=False,
                            handle_unknown='ignore',
                            drop='first'  # Avoid multicollinearity
                        )
                        self.encoders_[col].fit(X[[col]])
                    else:  # label encoding
                        self.encoders_[col] = LabelEncoder()
                        self.encoders_[col].fit(X[col].astype(str))
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in self.columns:
                if col in X.columns and col in self.encoders_:
                    if self.encoding_type == 'onehot':
                        encoded = self.encoders_[col].transform(X[[col]])
                        # Create column names
                        feature_names = [f"{col}_{cat}" for cat in self.encoders_[col].categories_[0][1:]]
                        encoded_df = pd.DataFrame(
                            encoded,
                            columns=feature_names,
                            index=X.index
                        )
                        # Drop original column and add encoded
                        X = X.drop(columns=[col])
                        X = pd.concat([X, encoded_df], axis=1)
                    else:  # label encoding
                        X[f"{col}_encoded"] = self.encoders_[col].transform(X[col].astype(str))
                        X = X.drop(columns=[col])
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling missing values.
    Supports imputation (mean, median, mode, KNN) or removal.
    """
    
    def __init__(self, strategy: str = 'mean', columns: Optional[List[str]] = None):
        """
        Args:
            strategy: 'mean', 'median', 'mode', 'knn', or 'remove'
            columns: List of columns to handle (None = all numeric columns)
        """
        self.strategy = strategy
        self.columns = columns
        self.imputers_ = {}
        self.columns_to_drop_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit imputers or identify columns to drop."""
        if isinstance(X, pd.DataFrame):
            if self.columns is None:
                # Default to numeric columns
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                self.columns = numeric_cols
            
            if self.strategy == 'remove':
                # Identify columns with >50% missing values
                missing_pct = X[self.columns].isnull().mean()
                self.columns_to_drop_ = missing_pct[missing_pct > 0.5].index.tolist()
            else:
                # Fit imputers
                for col in self.columns:
                    if col in X.columns:
                        if self.strategy == 'knn':
                            self.imputers_[col] = KNNImputer(n_neighbors=5)
                        else:
                            self.imputers_[col] = SimpleImputer(strategy=self.strategy)
                        self.imputers_[col].fit(X[[col]])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing or removing missing values."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            
            if self.strategy == 'remove':
                # Drop columns with high missing values
                X = X.drop(columns=self.columns_to_drop_)
                # Drop rows with any remaining missing values
                X = X.dropna()
            else:
                # Impute missing values
                for col in self.columns:
                    if col in X.columns and col in self.imputers_:
                        X[[col]] = self.imputers_[col].transform(X[[col]])
        return X


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for Weight of Evidence (WoE) transformation.
    Uses xverse library if available, otherwise implements basic WoE.
    """
    
    def __init__(self, columns: List[str], target_col: str = 'is_high_risk'):
        """
        Args:
            columns: List of categorical columns to transform
            target_col: Target variable name for WoE calculation
        """
        self.columns = columns
        self.target_col = target_col
        self.woe_encoders_ = {}
        self.iv_scores_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Calculate WoE and IV for each categorical column."""
        if isinstance(X, pd.DataFrame):
            # Use y if provided, otherwise use target_col from X
            target = y if y is not None else X[self.target_col] if self.target_col in X.columns else None
            
            if target is not None:
                for col in self.columns:
                    if col in X.columns:
                        # Calculate WoE and IV
                        woe_dict, iv = self._calculate_woe_iv(X[col], target)
                        self.woe_encoders_[col] = woe_dict
                        self.iv_scores_[col] = iv
        return self
    
    def _calculate_woe_iv(self, feature: pd.Series, target: pd.Series) -> Tuple[Dict, float]:
        """Calculate WoE and IV for a feature."""
        # Create cross-tabulation
        crosstab = pd.crosstab(feature, target, margins=False)
        
        # Calculate percentages
        total_good = (target == 0).sum()
        total_bad = (target == 1).sum()
        
        woe_dict = {}
        iv = 0.0
        
        for category in crosstab.index:
            good_count = crosstab.loc[category, 0] if 0 in crosstab.columns else 0
            bad_count = crosstab.loc[category, 1] if 1 in crosstab.columns else 0
            
            # Avoid division by zero
            if good_count == 0:
                good_pct = 0.0001
            else:
                good_pct = good_count / total_good
            
            if bad_count == 0:
                bad_pct = 0.0001
            else:
                bad_pct = bad_count / total_bad
            
            # Calculate WoE
            if bad_pct == 0:
                woe = 0
            else:
                woe = np.log(good_pct / bad_pct)
            
            woe_dict[category] = woe
            
            # Calculate IV contribution
            iv += (good_pct - bad_pct) * woe
        
        return woe_dict, iv
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using WoE values."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in self.columns:
                if col in X.columns and col in self.woe_encoders_:
                    # Replace categories with WoE values
                    X[f"{col}_woe"] = X[col].map(self.woe_encoders_[col]).fillna(0)
                    # Drop original column
                    X = X.drop(columns=[col])
        return X
    
    def get_iv_scores(self) -> Dict[str, float]:
        """Get Information Value scores for all features."""
        return self.iv_scores_


class CreditRiskDataProcessor:
    """
    Data processor for credit risk modeling using sklearn Pipeline.
    Handles feature engineering, RFM analysis, and proxy variable creation.
    """
    
    def __init__(self, use_woe: bool = True, encoding_type: str = 'onehot'):
        """
        Initialize the data processor.
        
        Args:
            use_woe: Whether to use WoE transformation for categorical features
            encoding_type: 'onehot' or 'label' for categorical encoding
        """
        self.use_woe = use_woe
        self.encoding_type = encoding_type
        self.feature_columns = None
        self.pipeline = None
        self.scaler = None
        self.target_col = 'is_high_risk'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load transaction data from CSV file."""
        df = pd.read_csv(file_path)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        return df
    
    def create_proxy_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create proxy variable for credit risk (high risk = 1, low risk = 0).
        Uses RFM patterns and fraud indicators.
        """
        # Calculate customer-level features
        customer_features = self._calculate_customer_features(df)
        
        # Define high risk criteria
        has_fraud = customer_features['has_fraud'] == 1
        low_engagement = (customer_features['recency_days'] > 30) & (customer_features['transaction_count'] < 5)
        high_volatility = (customer_features['amount_std'] > customer_features['amount_mean']) & (customer_features['amount_mean'] < 1000)
        declining_pattern = customer_features['recent_transaction_ratio'] < 0.5
        
        # Combine criteria
        customer_features['is_high_risk'] = (
            has_fraud | low_engagement | high_volatility | declining_pattern
        ).astype(int)
        
        # Merge back to transaction level
        df = df.merge(
            customer_features[['CustomerId', 'is_high_risk']],
            on='CustomerId',
            how='left'
        )
        df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)
        
        return df
    
    def _calculate_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer-level features for RFM analysis."""
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
        customer_features['has_fraud'] = (customer_features['fraud_count'] > 0).astype(int)
        customer_features['transaction_span_days'] = (
            customer_features['last_transaction'] - customer_features['first_transaction']
        ).dt.days
        customer_features['avg_days_between_transactions'] = (
            customer_features['transaction_span_days'] / 
            (customer_features['transaction_count'] - 1).replace(0, 1)
        )
        
        # Recent transaction ratio
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
    
    def build_pipeline(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        scaling_method: str = 'standard'
    ) -> Pipeline:
        """
        Build sklearn Pipeline with all transformation steps.
        
        Args:
            categorical_cols: List of categorical columns to encode
            numerical_cols: List of numerical columns to scale
            scaling_method: 'standard', 'minmax', or 'robust'
            
        Returns:
            sklearn Pipeline object
        """
        # Default categorical columns
        if categorical_cols is None:
            categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'CurrencyCode']
        
        # Default numerical columns (will be determined after feature engineering)
        if numerical_cols is None:
            numerical_cols = []
        
        # Build pipeline steps
        steps = []
        
        # Step 1: Temporal features (extract first, before aggregations)
        steps.append(('temporal', TemporalFeatureExtractor()))
        
        # Step 2: Aggregate features (creates customer-level aggregations)
        steps.append(('aggregate', AggregateFeatureTransformer()))
        
        # Step 3: Handle missing values
        steps.append(('missing_values', MissingValueHandler(strategy='mean')))
        
        # Step 4: Categorical encoding
        if self.use_woe and (XVERSE_AVAILABLE or WOE_LIB_AVAILABLE):
            # Use WoE transformation
            steps.append(('woe', WOETransformer(columns=categorical_cols)))
        else:
            # Use One-Hot or Label encoding
            steps.append(('categorical', CategoricalEncoder(
                columns=categorical_cols,
                encoding_type=self.encoding_type
            )))
        
        # Step 5: Scaling (will be applied separately to numerical features)
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:  # robust
            scaler = RobustScaler()
        
        self.scaler = scaler
        
        # Create pipeline
        self.pipeline = Pipeline(steps)
        
        return self.pipeline
    
    def process_data(
        self,
        file_path: str,
        is_training: bool = True,
        scaling_method: str = 'standard'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete data processing pipeline using sklearn Pipeline.
        
        Args:
            file_path: Path to CSV file
            is_training: Whether this is training data
            scaling_method: 'standard', 'minmax', or 'robust'
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Load data
        df = self.load_data(file_path)
        
        # Create proxy variable if training
        if is_training:
            df = self.create_proxy_variable(df)
        
        # Calculate additional customer features (RFM, etc.) and merge
        # Note: Aggregate features (total_amount, avg_amount, etc.) will be created by pipeline
        customer_features = self._calculate_customer_features(df)
        # Remove columns that will be created by aggregate transformer to avoid conflicts
        customer_features = customer_features.drop(
            columns=['amount_sum', 'amount_mean', 'amount_std', 'transaction_count'],
            errors='ignore'
        )
        df = df.merge(customer_features, on='CustomerId', how='left')
        
        # Identify categorical and numerical columns
        categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'CurrencyCode']
        
        # Build pipeline
        if self.pipeline is None:
            self.build_pipeline(
                categorical_cols=categorical_cols,
                scaling_method=scaling_method
            )
        
        # Apply pipeline transformations
        if is_training:
            # Fit and transform
            df_transformed = self.pipeline.fit_transform(df)
        else:
            # Only transform
            df_transformed = self.pipeline.transform(df)
        
        # Get WoE IV scores if available
        if self.use_woe and hasattr(self.pipeline.named_steps.get('woe'), 'get_iv_scores'):
            iv_scores = self.pipeline.named_steps['woe'].get_iv_scores()
            print("\nInformation Value (IV) Scores:")
            for col, iv in iv_scores.items():
                print(f"  {col}: {iv:.4f}")
        
        # Separate features and target
        if is_training:
            # Drop ID columns and target
            feature_cols = [col for col in df_transformed.columns 
                          if col not in ['CustomerId', 'AccountId', 'SubscriptionId', 
                                        'TransactionId', 'BatchId', self.target_col]]
            X = df_transformed[feature_cols]
            y = df_transformed[self.target_col] if self.target_col in df_transformed.columns else None
        else:
            feature_cols = [col for col in df_transformed.columns 
                          if col not in ['CustomerId', 'AccountId', 'SubscriptionId', 
                                        'TransactionId', 'BatchId']]
            X = df_transformed[feature_cols]
            y = None
        
        # Apply scaling to numerical features
        if self.scaler is not None:
            numerical_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerical_feature_cols) > 0:
                # Create copy to avoid SettingWithCopyWarning
                X = X.copy()
                if is_training:
                    X.loc[:, numerical_feature_cols] = self.scaler.fit_transform(X[numerical_feature_cols])
                else:
                    X.loc[:, numerical_feature_cols] = self.scaler.transform(X[numerical_feature_cols])
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_columns if self.feature_columns else []
    
    def process_data_from_df(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        scaling_method: str = 'standard'
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Process data from DataFrame (for inference).
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data
            scaling_method: 'standard', 'minmax', or 'robust'
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        # Ensure TransactionStartTime is datetime
        if 'TransactionStartTime' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['TransactionStartTime']):
                df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Calculate customer features and merge
        customer_features = self._calculate_customer_features(df)
        df = df.merge(customer_features, on='CustomerId', how='left')
        
        # Build pipeline if not already built
        if self.pipeline is None:
            categorical_cols = ['ProductCategory', 'ChannelId', 'ProviderId', 'CurrencyCode']
            self.build_pipeline(
                categorical_cols=categorical_cols,
                scaling_method=scaling_method
            )
        
        # Apply pipeline transformations
        if is_training:
            df_transformed = self.pipeline.fit_transform(df)
        else:
            df_transformed = self.pipeline.transform(df)
        
        # Separate features and target
        feature_cols = [col for col in df_transformed.columns 
                      if col not in ['CustomerId', 'AccountId', 'SubscriptionId', 
                                    'TransactionId', 'BatchId', self.target_col]]
        X = df_transformed[feature_cols]
        
        if is_training and self.target_col in df_transformed.columns:
            y = df_transformed[self.target_col]
        else:
            y = None
        
        # Apply scaling
        if self.scaler is not None:
            numerical_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerical_feature_cols) > 0:
                # Create copy to avoid SettingWithCopyWarning
                X = X.copy()
                if is_training:
                    X.loc[:, numerical_feature_cols] = self.scaler.fit_transform(X[numerical_feature_cols])
                else:
                    X.loc[:, numerical_feature_cols] = self.scaler.transform(X[numerical_feature_cols])
        
        # Store feature columns
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        return X, y
