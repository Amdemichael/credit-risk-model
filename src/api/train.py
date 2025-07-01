import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score,
                           precision_score, recall_score, f1_score)
from pathlib import Path
import os
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import warnings

# Disable warnings if needed (updated for newer MLflow)
warnings.filterwarnings("ignore", category=UserWarning)

def get_project_root():
    """Returns absolute path to project root with Windows compatibility"""
    return Path(__file__).parent.parent.parent

def setup_mlflow():
    """Configure MLflow tracking with Windows-compatible paths"""
    project_root = get_project_root()
    mlruns_dir = project_root / "mlruns"
    
    # Ensure directory exists with proper permissions
    mlruns_dir.mkdir(exist_ok=True)
    
    # Convert to URI with forward slashes
    tracking_uri = "file:///" + str(mlruns_dir.absolute()).replace('\\', '/')
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        mlflow.set_experiment("CreditRisk")
    except Exception as e:
        print(f"Experiment setup note: {str(e)}")

def convert_data_types(df):
    """Convert integer columns to float to avoid MLflow warnings"""
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('float64')
    return df

def train_model():
    # Get absolute path to data
    project_root = get_project_root()
    data_path = project_root / "Data" / "processed" / "model_features_with_target.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = convert_data_types(df)
    X = df.drop(columns=["is_high_risk", "customerid"])
    y = df["is_high_risk"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    setup_mlflow()
    
    with mlflow.start_run(run_name="RF_Production") as run:
        # Model configuration
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        
        # Training
        model = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred))
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.best_params_)
        
        # Model signature and input example
        signature = infer_signature(X_train, y_train)
        
        # Log model        
        mlflow.sklearn.log_model(
            sk_model=model.best_estimator_,
            name="model",  # New parameter name
            registered_model_name="CreditRiskModel",
            signature=signature,
            input_example=X_train.iloc[:1]
        )

if __name__ == "__main__":
    train_model()