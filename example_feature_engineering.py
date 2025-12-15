"""
Example script demonstrating Task 3 - Feature Engineering Pipeline
Shows how to use the sklearn Pipeline for feature engineering.
"""

import sys
sys.path.append('.')

from src.data_processing import CreditRiskDataProcessor
import pandas as pd

def main():
    """Demonstrate feature engineering pipeline."""
    print("="*80)
    print("TASK 3: FEATURE ENGINEERING PIPELINE EXAMPLE")
    print("="*80)
    
    # Initialize processor
    print("\n[1] Initializing CreditRiskDataProcessor...")
    processor = CreditRiskDataProcessor(use_woe=False, encoding_type='onehot')
    print("  Processor initialized")
    
    # Process data with pipeline
    print("\n[2] Processing data through sklearn Pipeline...")
    print("  Pipeline steps:")
    print("    1. Temporal Feature Extraction (hour, day, month, year)")
    print("    2. Aggregate Features (total_amount, avg_amount, transaction_count, amount_std)")
    print("    3. Missing Value Handling (mean imputation)")
    print("    4. Categorical Encoding (One-Hot Encoding)")
    print("    5. Scaling (StandardScaler)")
    
    X, y = processor.process_data(
        'data/raw/data.csv',
        is_training=True,
        scaling_method='standard'
    )
    
    print(f"\n[3] Results:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Number of features: {len(X.columns)}")
    
    # Show feature types
    print(f"\n[4] Feature Breakdown:")
    aggregate_features = [col for col in X.columns if any(x in col for x in ['total_amount', 'avg_amount', 'transaction_count', 'amount_std'])]
    temporal_features = [col for col in X.columns if any(x in col for x in ['hour', 'day', 'month', 'year'])]
    categorical_features = [col for col in X.columns if any(x in col for x in ['ProductCategory', 'ChannelId', 'ProviderId'])]
    
    print(f"  Aggregate features: {len(aggregate_features)}")
    print(f"    Examples: {aggregate_features[:4]}")
    print(f"  Temporal features: {len(temporal_features)}")
    print(f"    Examples: {temporal_features[:4]}")
    print(f"  Categorical (encoded) features: {len(categorical_features)}")
    print(f"    Examples: {categorical_features[:5]}")
    
    # Show pipeline structure
    print(f"\n[5] Pipeline Structure:")
    if processor.pipeline:
        for i, (name, transformer) in enumerate(processor.pipeline.steps, 1):
            print(f"    Step {i}: {name} - {transformer.__class__.__name__}")
    
    # Show scaling info
    if processor.scaler:
        print(f"\n[6] Scaling Applied:")
        print(f"    Scaler type: {processor.scaler.__class__.__name__}")
        print(f"    Numerical features scaled: {len(X.select_dtypes(include=['float64', 'int64']).columns)}")
    
    print("\n" + "="*80)
    print("Feature Engineering Pipeline Complete!")
    print("="*80)
    print("\nThe pipeline is:")
    print("  ✓ Automated - All transformations chained together")
    print("  ✓ Reproducible - Same transformations applied consistently")
    print("  ✓ Robust - Handles missing values and new categories")
    print("  ✓ Ready for model training")

if __name__ == "__main__":
    main()

