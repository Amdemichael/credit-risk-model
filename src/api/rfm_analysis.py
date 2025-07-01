"""
Standalone script for Task 4: RFM Analysis and Proxy Target Creation
Creates a binary 'is_high_risk' label based on customer transaction behavior
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFMAnalyzer:
    """
    Performs RFM analysis and creates high-risk proxy variable
    """
    def __init__(self, n_clusters: int = 3, snapshot_date: str = None, random_state: int = 42):
        """
        Args:
            n_clusters: Number of clusters for segmentation
            snapshot_date: Reference date for recency calculation (YYYY-MM-DD)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else pd.Timestamp.now()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.high_risk_cluster_ = None

    def calculate_rfm(self, df: pd.DataFrame, customer_id_col: str = 'customerid', 
                     date_col: str = 'transactionstarttime', value_col: str = 'value') -> pd.DataFrame:
        """
        Calculate RFM metrics from transaction data
        Args:
            df: Raw transaction data
            customer_id_col: Name of customer ID column
            date_col: Name of transaction date column
            value_col: Name of transaction value column
        Returns:
            DataFrame with RFM metrics per customer
        """
        # Ensure datetime format
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Calculate RFM metrics
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            customer_id_col: 'count',                                  # Frequency
            value_col: 'mean'                                         # Monetary
        }).rename(columns={
            date_col: 'recency',
            customer_id_col: 'frequency',
            value_col: 'monetary'
        }).reset_index()
        
        return rfm

    def identify_high_risk_cluster(self, rfm: pd.DataFrame) -> int:
        """
        Identify which cluster represents high-risk customers
        High risk = high recency, low frequency, low monetary value
        """
        cluster_stats = rfm.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary': 'mean'
        })
        # Sort by recency (desc), frequency (asc), monetary (asc)
        return cluster_stats.sort_values(
            ['recency', 'frequency', 'monetary'],
            ascending=[False, True, True]
        ).index[0]

    def create_proxy_target(self, transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main method that creates the proxy target variable
        Args:
            transactions_df: Raw transaction data
        Returns:
            Tuple of (RFM metrics with clusters, original data with is_high_risk column)
        """
        # Calculate RFM metrics
        rfm_metrics = self.calculate_rfm(transactions_df)
        
        # Scale features
        rfm_scaled = self.scaler.fit_transform(rfm_metrics[['recency', 'frequency', 'monetary']])
        
        # Cluster customers
        rfm_metrics['cluster'] = self.kmeans.fit_predict(rfm_scaled)
        
        # Identify high-risk cluster
        self.high_risk_cluster_ = self.identify_high_risk_cluster(rfm_metrics)
        rfm_metrics['is_high_risk'] = (rfm_metrics['cluster'] == self.high_risk_cluster_).astype(int)
        
        # Merge back with original data
        enriched_df = transactions_df.merge(
            rfm_metrics[['customerid', 'is_high_risk', 'cluster']],
            on='customerid',
            how='left'
        )
        
        return rfm_metrics, enriched_df

    def plot_rfm_clusters(self, rfm: pd.DataFrame, save_path: str = None) -> None:
        """
        Visualize RFM clusters
        Args:
            rfm: DataFrame with RFM metrics and cluster assignments
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(15, 5))
        
        # Create subplots
        plt.subplot(131)
        sns.scatterplot(data=rfm, x='recency', y='frequency', hue='cluster', palette='viridis')
        plt.title('Recency vs Frequency')
        
        plt.subplot(132)
        sns.scatterplot(data=rfm, x='frequency', y='monetary', hue='cluster', palette='viridis')
        plt.title('Frequency vs Monetary')
        
        plt.subplot(133)
        sns.scatterplot(data=rfm, x='recency', y='monetary', hue='cluster', palette='viridis')
        plt.title('Recency vs Monetary')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved RFM cluster plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

def run_rfm_analysis(input_path: str, output_path: str, plot_path: str = None) -> None:
    """
    Complete RFM analysis workflow
    Args:
        input_path: Path to raw transaction data
        output_path: Path to save enriched data with proxy target
        plot_path: Optional path to save RFM cluster visualization
    """
    # Load data
    logger.info(f"Loading data from {input_path}")
    transactions = pd.read_csv(input_path)
    
    # Initialize and run RFM analysis
    analyzer = RFMAnalyzer(n_clusters=3, snapshot_date=transactions['transactionstarttime'].max())
    rfm_metrics, enriched_data = analyzer.create_proxy_target(transactions)
    
    # Save results
    enriched_data.to_csv(output_path, index=False)
    logger.info(f"Saved enriched data with proxy target to {output_path}")
    
    # Visualize clusters
    if plot_path:
        analyzer.plot_rfm_clusters(rfm_metrics, plot_path)
    
    # Print summary
    risk_distribution = enriched_data['is_high_risk'].value_counts(normalize=True)
    logger.info("\nProxy Target Distribution:")
    logger.info(f"Low Risk (0): {risk_distribution.get(0, 0):.2%}")
    logger.info(f"High Risk (1): {risk_distribution.get(1, 0):.2%}")

if __name__ == "__main__":
    # Configuration
    INPUT_DATA = "../data/raw/transactions.csv"
    OUTPUT_DATA = "../data/processed/enriched_transactions.csv"
    PLOT_PATH = "../reports/figures/rfm_clusters.png"
    
    # Run analysis
    run_rfm_analysis(
        input_path=INPUT_DATA,
        output_path=OUTPUT_DATA,
        plot_path=PLOT_PATH
    )