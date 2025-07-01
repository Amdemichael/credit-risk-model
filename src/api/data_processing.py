# src/data_processing.py
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
import os
from pathlib import Path

### ─── Data Structures ─────────────────────────────────────

@dataclass
class DataPaths:
    raw: str = '../data/raw/data.csv'
    processed: str = '../data/processed/features.csv'
    feature_mapping: str = '../data/processed/feature_mapping.txt'

### ─── Core Transformers ───────────────────────────────────
class DataLoader(BaseEstimator, TransformerMixin):
    """Handles data loading and initial cleaning with feature names support"""
    def __init__(self, file_path: str = DataPaths.raw):
        self.file_path = file_path
        self.feature_names_ = None
        
    def fit(self, X=None, y=None):
        # Load sample data to determine feature names
        sample_df = pd.read_csv(self.file_path, nrows=1)
        sample_df = self._clean_column_names(sample_df)
        sample_df = self._convert_datetime(sample_df)
        self.feature_names_ = list(sample_df.columns)
        return self
        
    def transform(self, X=None, y=None) -> pd.DataFrame:
        """Load and clean data"""
        df = pd.read_csv(self.file_path)
        return (
            df.pipe(self._clean_column_names)
              .pipe(self._convert_datetime)
        )
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Required for scikit-learn feature names support"""
        return self.feature_names_
    
    @staticmethod
    def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    
    @staticmethod
    def _convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df['transactionstarttime'] = pd.to_datetime(df['transactionstarttime'], errors='coerce')
        return df

class RFMFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generates RFM features with proper timezone handling"""
    def __init__(self, n_clusters: int = 3, snapshot_date: Optional[str] = None):
        self.n_clusters = n_clusters
        self.snapshot_date = self._ensure_naive_datetime(
            pd.to_datetime(snapshot_date) if snapshot_date else pd.Timestamp.now()
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        rfm_metrics = self._calculate_rfm_metrics(X)
        self.kmeans.fit(rfm_metrics[['recency', 'frequency', 'monetary']])
        
        clustered = rfm_metrics.assign(
            cluster=self.kmeans.predict(rfm_metrics[['recency', 'frequency', 'monetary']])
        )
        self.high_risk_cluster_ = self._identify_high_risk_cluster(clustered)
        self.feature_names_ = list(X.columns) + ['rfm_cluster', 'is_high_risk']
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        rfm_metrics = self._calculate_rfm_metrics(X)
        rfm_metrics['rfm_cluster'] = self.kmeans.predict(rfm_metrics[['recency', 'frequency', 'monetary']])
        rfm_metrics['is_high_risk'] = (rfm_metrics['rfm_cluster'] == self.high_risk_cluster_).astype(int)
        return X.merge(rfm_metrics, on='customerid', how='left')
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_
    
    def _calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if hasattr(df['transactionstarttime'].dtype, 'tz'):
            df['transactionstarttime'] = df['transactionstarttime'].dt.tz_localize(None)
        return df.groupby('customerid').agg(
            recency=('transactionstarttime', lambda x: (self.snapshot_date - x.max()).days),
            frequency=('transactionid', 'count'),
            monetary=('value', 'mean')
        ).reset_index()
    
    def _identify_high_risk_cluster(self, rfm: pd.DataFrame) -> int:
        cluster_stats = rfm.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary': 'mean'
        })
        return cluster_stats.sort_values(
            ['recency', 'frequency', 'monetary'],
            ascending=[False, True, True]
        ).index[0]
    
    @staticmethod
    def _ensure_naive_datetime(dt: pd.Timestamp) -> pd.Timestamp:
        if hasattr(dt, 'tz') and dt.tz is not None:
            return dt.tz_localize(None)
        return dt

class BehavioralFeatureGenerator(BaseEstimator, TransformerMixin):
    """Creates behavioral transaction patterns"""
    def __init__(self):
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        self.feature_names_ = list(X.columns) + ['days_since_last', 'value_zscore', 'category_change_count']
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.sort_values(['customerid', 'transactionstarttime'])
        return df.assign(
            days_since_last=df.groupby('customerid')['transactionstarttime'].diff().dt.days,
            value_zscore=df.groupby('customerid')['value'].transform(
                lambda x: (x - x.mean()) / x.std()),
            category_change_count=df.groupby('customerid')['productcategory'].transform(
                lambda x: x.ne(x.shift()).cumsum())
        )
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    """Extracts and encodes temporal features"""
    def __init__(self):
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        self.feature_names_ = list(X.columns) + ['hour_sin', 'hour_cos', 'weekday_sin']
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dt_col = X['transactionstarttime']
        return X.assign(
            hour_sin=np.sin(2 * np.pi * dt_col.dt.hour/24),
            hour_cos=np.cos(2 * np.pi * dt_col.dt.hour/24),
            weekday_sin=np.sin(2 * np.pi * dt_col.dt.weekday/7)
        )
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

### ─── Pipeline Construction ───────────────────────────────

def create_feature_pipeline() -> Pipeline:
    """Constructs the complete feature engineering pipeline"""
    numerical_features = ['amount', 'value', 'recency', 'frequency', 'monetary']
    categorical_features = ['productcategory', 'channelid']
    time_features = ['hour_sin', 'hour_cos', 'weekday_sin']
    behavioral_features = ['days_since_last', 'value_zscore', 'category_change_count']
    
    # Create transformers with feature names support
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Modified preprocessor to preserve customerid
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('time', 'passthrough', time_features),
        ('behavioral', 'passthrough', behavioral_features),
        ('id', 'passthrough', ['customerid'])  # Explicitly pass through customerid
    ])
    
    return Pipeline([
        ('loader', DataLoader()),
        ('rfm', RFMFeatureGenerator()),
        ('behavior', BehavioralFeatureGenerator()),
        ('time_features', TimeFeatureGenerator()),
        ('preprocessor', preprocessor)
    ])

### ─── Main Processing Function ────────────────────────────

def process_data(
    input_path: str = DataPaths.raw,
    output_path: str = DataPaths.processed,
    mapping_path: str = DataPaths.feature_mapping
) -> pd.DataFrame:
    """
    Complete data processing workflow
    Args:
        input_path: Path to raw data file
        output_path: Path to save processed data
        mapping_path: Path to save feature mapping
    Returns:
        Processed DataFrame
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = create_feature_pipeline()
    processed_data = pipeline.fit_transform(None)
    
    # Get feature names with fallback
    try:
        feature_names = pipeline.get_feature_names_out()
        if feature_names is None:
            raise AttributeError("Feature names not available")
    except (AttributeError, NotImplementedError):
        num_features = processed_data.shape[1]
        feature_names = [f"feature_{i}" for i in range(num_features)]
        print("⚠️ Could not get feature names - using generated column names")
    
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    
    # Save data and feature mapping
    processed_df.to_csv(output_path, index=False)
    with open(mapping_path, 'w') as f:
        f.write("Feature Mapping:\n")
        f.write("\n".join(f"{i}: {name}" for i, name in enumerate(feature_names)))
    
    print(f"✅ Processed data saved to {output_path}")
    print(f"✅ Feature mapping saved to {mapping_path}")
    print(f"Final feature matrix shape: {processed_df.shape}")
    
    return processed_df

### ─── EDA Functions ──────────────────────────────────────

def get_summary_statistics(df: pd.DataFrame):
    """Original EDA function preserved"""
    print("🔍 Summary Statistics:\n", df.describe())
    print("\n🔍 Missing Values:\n", df.isnull().sum())
    print("\n🔍 Data Types:\n", df.dtypes)