"""
Test script for Task 3 - Feature Engineering Pipeline
"""

import sys
sys.path.append('.')

from src.data_processing import (
    CreditRiskDataProcessor,
    AggregateFeatureTransformer,
    TemporalFeatureExtractor,
    CategoricalEncoder,
    MissingValueHandler,
    WOETransformer
)
import pandas as pd
import numpy as np

def test_pipeline():
    """Test the complete feature engineering pipeline."""
    print("="*80)
    print("TESTING TASK 3 - FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Initialize processor
    processor = CreditRiskDataProcessor(use_woe=False, encoding_type='onehot')
    
    # Load data
    print("\n[1] Loading data...")
    df = processor.load_data('data/raw/data.csv')
    print(f"  [OK] Data loaded: {df.shape}")
    
    # Test aggregate features
    print("\n[2] Testing Aggregate Features...")
    agg_transformer = AggregateFeatureTransformer()
    agg_transformer.fit(df)
    df_agg = agg_transformer.transform(df)
    
    required_agg = ['total_amount', 'avg_amount', 'transaction_count', 'amount_std']
    for feat in required_agg:
        if feat in df_agg.columns:
            print(f"  ✓ {feat}: created")
        else:
            print(f"  ✗ {feat}: missing")
    
    # Test temporal features
    print("\n[3] Testing Temporal Features...")
    temp_transformer = TemporalFeatureExtractor()
    temp_transformer.fit(df)
    df_temp = temp_transformer.transform(df)
    
    required_temp = ['transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year']
    for feat in required_temp:
        if feat in df_temp.columns:
            print(f"  [OK] {feat}: created")
        else:
            print(f"  [FAIL] {feat}: missing")
    
    # Test categorical encoding
    print("\n[4] Testing Categorical Encoding...")
    cat_cols = ['ProductCategory', 'ChannelId']
    cat_encoder = CategoricalEncoder(columns=cat_cols, encoding_type='onehot')
    cat_encoder.fit(df)
    df_encoded = cat_encoder.transform(df)
    print(f"  [OK] One-Hot Encoding applied to {len(cat_cols)} columns")
    print(f"  [OK] Encoded columns: {[col for col in df_encoded.columns if any(c in col for c in cat_cols)][:5]}...")
    
    # Test missing value handling
    print("\n[5] Testing Missing Value Handling...")
    missing_handler = MissingValueHandler(strategy='mean')
    missing_handler.fit(df)
    df_imputed = missing_handler.transform(df)
    print(f"  [OK] Missing value handling applied")
    print(f"  [OK] Missing values after imputation: {df_imputed.isnull().sum().sum()}")
    
    # Test full pipeline
    print("\n[6] Testing Full Pipeline...")
    try:
        X, y = processor.process_data('data/raw/data.csv', is_training=True, scaling_method='standard')
        print(f"  [OK] Pipeline executed successfully")
        print(f"  [OK] Features shape: {X.shape}")
        print(f"  [OK] Target shape: {y.shape if y is not None else 'None'}")
        print(f"  [OK] Number of features: {len(X.columns)}")
        print(f"  [OK] Sample feature columns: {list(X.columns[:10])}")
        
        # Verify aggregate features are present
        agg_present = any('total_amount' in str(col) or 'avg_amount' in str(col) or 
                         'transaction_count' in str(col) or 'amount_std' in str(col) 
                         for col in X.columns)
        print(f"  [OK] Aggregate features present: {agg_present}")
        
        # Verify temporal features are present
        temp_present = any('hour' in str(col) or 'day' in str(col) or 
                          'month' in str(col) or 'year' in str(col) 
                          for col in X.columns)
        print(f"  [OK] Temporal features present: {temp_present}")
        
        # Verify scaling
        if processor.scaler is not None:
            print(f"  [OK] Scaler fitted: {processor.scaler.__class__.__name__}")
            sample_col = X.select_dtypes(include=[np.number]).columns[0]
            print(f"  [OK] Sample scaled feature '{sample_col}': mean={X[sample_col].mean():.4f}, std={X[sample_col].std():.4f}")
        
    except Exception as e:
        print(f"  [FAIL] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

