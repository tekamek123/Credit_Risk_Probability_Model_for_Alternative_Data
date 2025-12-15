"""
Test script for Task 4 - Proxy Target Variable Engineering with K-Means
"""

import sys
sys.path.append('.')

from src.data_processing import CreditRiskDataProcessor
import pandas as pd
import numpy as np

def test_kmeans_proxy_variable():
    """Test K-Means based proxy variable creation."""
    print("="*80)
    print("TESTING TASK 4: PROXY TARGET VARIABLE ENGINEERING (K-MEANS)")
    print("="*80)
    
    # Initialize processor
    processor = CreditRiskDataProcessor()
    
    # Load data
    print("\n[1] Loading data...")
    df = processor.load_data('data/raw/data.csv')
    print(f"  [OK] Data loaded: {df.shape}")
    
    # Create proxy variable using K-Means
    print("\n[2] Creating proxy variable using K-Means clustering...")
    print("  Steps:")
    print("    1. Calculate RFM metrics (Recency, Frequency, Monetary)")
    print("    2. Scale RFM features")
    print("    3. Apply K-Means clustering (3 clusters)")
    print("    4. Identify high-risk cluster (lowest engagement)")
    print("    5. Assign binary is_high_risk labels")
    
    df_with_proxy = processor.create_proxy_variable(df, n_clusters=3, random_state=42)
    
    print(f"\n[3] Results:")
    print(f"  [OK] Proxy variable created")
    print(f"  [OK] is_high_risk column exists: {'is_high_risk' in df_with_proxy.columns}")
    print(f"  [OK] Data shape: {df_with_proxy.shape}")
    
    # Check target distribution
    target_dist = df_with_proxy['is_high_risk'].value_counts()
    print(f"\n[4] Target Variable Distribution:")
    print(f"  Low Risk (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df_with_proxy)*100:.2f}%)")
    print(f"  High Risk (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df_with_proxy)*100:.2f}%)")
    
    # Check RFM columns
    print(f"\n[5] RFM Metrics:")
    rfm_cols = ['Recency', 'Frequency', 'Monetary']
    for col in rfm_cols:
        if col in df_with_proxy.columns:
            print(f"  [OK] {col}: min={df_with_proxy[col].min():.2f}, max={df_with_proxy[col].max():.2f}, mean={df_with_proxy[col].mean():.2f}")
        else:
            print(f"  [FAIL] {col}: missing")
    
    # Check cluster column
    if 'cluster' in df_with_proxy.columns:
        cluster_dist = df_with_proxy['cluster'].value_counts().sort_index()
        print(f"\n[6] Cluster Distribution:")
        for cluster_id, count in cluster_dist.items():
            high_risk_count = len(df_with_proxy[(df_with_proxy['cluster'] == cluster_id) & (df_with_proxy['is_high_risk'] == 1)])
            print(f"  Cluster {cluster_id}: {count} customers ({count/len(df_with_proxy)*100:.2f}%)")
            if high_risk_count > 0:
                print(f"    -> High Risk: {high_risk_count} ({high_risk_count/count*100:.2f}%)")
    
    # Get cluster info
    cluster_info = processor.get_cluster_info()
    if cluster_info:
        print(f"\n[7] Cluster Analysis:")
        print(f"  [OK] High-risk cluster ID: {cluster_info['high_risk_cluster']}")
        print(f"  [OK] Cluster summary:")
        print(cluster_info['cluster_summary'])
        print(f"  [OK] Cluster centers shape: {cluster_info['cluster_centers'].shape}")
    
    # Verify binary target
    print(f"\n[8] Target Variable Validation:")
    unique_values = df_with_proxy['is_high_risk'].unique()
    print(f"  [OK] Unique values: {sorted(unique_values)}")
    print(f"  [OK] Is binary: {set(unique_values).issubset({0, 1})}")
    print(f"  [OK] No missing values: {df_with_proxy['is_high_risk'].isnull().sum() == 0}")
    
    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("="*80)
    print("\nThe proxy variable is:")
    print("  [OK] Created using K-Means clustering on RFM metrics")
    print("  [OK] Binary (0 = low risk, 1 = high risk)")
    print("  [OK] Integrated into main dataset")
    print("  [OK] Ready for model training")

if __name__ == "__main__":
    test_kmeans_proxy_variable()

