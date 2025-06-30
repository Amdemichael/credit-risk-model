# src/target_engineering.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFMTargetEngineer:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.high_risk_cluster = None

    def calculate_rfm(self, processed_df):
        """Calculate RFM metrics from processed features"""
        # Group by customer using the preserved customerid
        rfm = processed_df.groupby('customerid').agg({
            'num__recency': 'first',
            'num__frequency': 'first',
            'num__monetary': 'first'
        }).reset_index()
        
        rfm.columns = ['customerid', 'recency', 'frequency', 'monetary']
        return rfm

    def create_proxy_target(self, processed_df):
        """Main method to create high-risk labels"""
        # Calculate RFM metrics
        rfm_metrics = self.calculate_rfm(processed_df)
        
        # Scale and cluster
        rfm_scaled = self.scaler.fit_transform(rfm_metrics[['recency', 'frequency', 'monetary']])
        rfm_metrics['cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Identify high-risk cluster
        cluster_stats = rfm_metrics.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).sort_values(
            ['recency', 'frequency', 'monetary'],
            ascending=[False, True, True]
        )
        self.high_risk_cluster = cluster_stats.index[0]
        rfm_metrics['is_high_risk'] = (rfm_metrics['cluster'] == self.high_risk_cluster).astype(int)
        
        # Merge back with processed data
        return processed_df.merge(
            rfm_metrics[['customerid', 'is_high_risk']],
            on='customerid',
            how='left'
        )

def run_rfm_analysis():
    """Complete workflow"""
    try:
        # File paths
        processed_path = "../data/processed/model_features.csv"
        output_path = "../data/processed/model_features_with_target.csv"
        plot_path = "../reports/figures/rfm_clusters.png"
        
        # Load data
        logger.info("Loading processed data")
        processed_df = pd.read_csv(processed_path)
        
        if 'customerid' not in processed_df.columns:
            raise ValueError("customerid column missing - please reprocess data with Task 3 fixes")
        
        # Create target variable
        engineer = RFMTargetEngineer()
        final_df = engineer.create_proxy_target(processed_df)
        
        # Save results
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
        
        # Print summary
        target_dist = final_df['is_high_risk'].value_counts(normalize=True)
        logger.info("\nTarget Distribution:")
        logger.info(f"Low Risk (0): {target_dist[0]:.2%}")
        logger.info(f"High Risk (1): {target_dist[1]:.2%}")
        
    except Exception as e:
        logger.error(f"Error in RFM analysis: {str(e)}")
        raise

if __name__ == "__main__":
    run_rfm_analysis()