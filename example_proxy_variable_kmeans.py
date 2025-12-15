"""
Example script demonstrating Task 4 - Proxy Target Variable Engineering with K-Means
Shows how to create a credit risk target variable using RFM metrics and K-Means clustering.
"""

import sys
sys.path.append('.')

from src.data_processing import CreditRiskDataProcessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Demonstrate K-Means based proxy variable creation."""
    print("="*80)
    print("TASK 4: PROXY TARGET VARIABLE ENGINEERING (K-MEANS CLUSTERING)")
    print("="*80)
    
    # Initialize processor
    print("\n[1] Initializing CreditRiskDataProcessor...")
    processor = CreditRiskDataProcessor()
    print("  Processor initialized")
    
    # Load data
    print("\n[2] Loading transaction data...")
    df = processor.load_data('data/raw/data.csv')
    print(f"  Data loaded: {df.shape}")
    print(f"  Unique customers: {df['CustomerId'].nunique()}")
    
    # Create proxy variable using K-Means
    print("\n[3] Creating proxy variable using K-Means clustering on RFM metrics...")
    print("  Process:")
    print("    1. Calculate RFM metrics for each customer")
    print("       - Recency: Days since last transaction")
    print("       - Frequency: Number of transactions")
    print("       - Monetary: Total transaction amount")
    print("    2. Scale RFM features (StandardScaler)")
    print("    3. Apply K-Means clustering (3 clusters, random_state=42)")
    print("    4. Identify high-risk cluster (lowest engagement)")
    print("    5. Assign binary is_high_risk labels")
    
    df_with_proxy = processor.create_proxy_variable(df, n_clusters=3, random_state=42)
    
    print(f"\n[4] Results:")
    print(f"  Data shape after adding proxy: {df_with_proxy.shape}")
    print(f"  New columns added: {set(df_with_proxy.columns) - set(df.columns)}")
    
    # Target distribution
    print(f"\n[5] Target Variable Distribution:")
    target_dist = df_with_proxy['is_high_risk'].value_counts()
    print(f"  Low Risk (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df_with_proxy)*100:.2f}%)")
    print(f"  High Risk (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df_with_proxy)*100:.2f}%)")
    
    # Cluster analysis
    cluster_info = processor.get_cluster_info()
    if cluster_info:
        print(f"\n[6] Cluster Analysis:")
        print(f"  High-risk cluster ID: {cluster_info['high_risk_cluster']}")
        print(f"\n  Cluster Summary (Average RFM values):")
        print(cluster_info['cluster_summary'])
        
        # Cluster distribution
        cluster_dist = df_with_proxy['cluster'].value_counts().sort_index()
        print(f"\n  Cluster Distribution:")
        for cluster_id in sorted(cluster_dist.index):
            count = cluster_dist[cluster_id]
            high_risk_count = len(df_with_proxy[
                (df_with_proxy['cluster'] == cluster_id) & 
                (df_with_proxy['is_high_risk'] == 1)
            ])
            print(f"    Cluster {cluster_id}: {count:,} customers ({count/len(df_with_proxy)*100:.2f}%)")
            if high_risk_count > 0:
                print(f"      -> High Risk: {high_risk_count:,} ({high_risk_count/count*100:.2f}%)")
    
    # RFM statistics
    print(f"\n[7] RFM Metrics Statistics:")
    rfm_stats = df_with_proxy[['Recency', 'Frequency', 'Monetary']].describe()
    print(rfm_stats)
    
    # High-risk vs Low-risk comparison
    print(f"\n[8] High-Risk vs Low-Risk Customer Comparison:")
    comparison = df_with_proxy.groupby('is_high_risk')[['Recency', 'Frequency', 'Monetary']].mean()
    print(comparison)
    
    print("\n" + "="*80)
    print("Proxy Variable Creation Complete!")
    print("="*80)
    print("\nKey Insights:")
    print("  ✓ High-risk customers identified using K-Means clustering")
    print("  ✓ Clustering based on engagement patterns (RFM metrics)")
    print("  ✓ Binary target variable (is_high_risk) ready for model training")
    print("  ✓ Reproducible with random_state parameter")
    print("  ✓ Integrated into main dataset for seamless model training")

if __name__ == "__main__":
    main()

