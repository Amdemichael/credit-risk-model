import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report)
import mlflow
import mlflow.sklearn
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load processed data with target variable"""
    data_path = "../data/processed/model_features_with_target.csv"
    df = pd.read_csv(data_path)
    
    # Verify required columns
    if 'is_high_risk' not in df.columns:
        raise ValueError("Target variable 'is_high_risk' not found in data")
    
    X = df.drop(columns=['is_high_risk', 'customerid'])
    y = df['is_high_risk']
    return X, y

def evaluate_model(model, X_test, y_test):
    """Calculate and log all required metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return metrics

def train_models():
    """Main training function with MLflow tracking"""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Initialize MLflow
    mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
    mlflow.set_experiment("credit_risk_modeling")
    
    # Model candidates
    models = {
        "logistic_regression": {
            "model": LogisticRegression(class_weight='balanced', random_state=42),
            "params": {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            }
        },
        "random_forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20]
            }
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1]
            }
        }
    }
    
    best_score = 0
    best_model = None
    
    for model_name, config in models.items():
        with mlflow.start_run(run_name=model_name):
            # Hyperparameter tuning
            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=5,
                scoring='roc_auc',
                cv=5,
                random_state=42
            )
            search.fit(X_train, y_train)
            
            # Evaluate
            metrics = evaluate_model(search.best_estimator_, X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, model_name)
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = search.best_estimator_
                best_model_name = model_name
    
    # Register best model
    if best_model:
        mlflow.sklearn.log_model(best_model, "best_model")
        logger.info(f"\nBest model: {best_model_name} with ROC-AUC: {best_score:.4f}")
        
        # Save for deployment
        Path("../models").mkdir(exist_ok=True)
        mlflow.sklearn.save_model(best_model, "../models/best_model")
        
    return best_model

if __name__ == "__main__":
    train_models()