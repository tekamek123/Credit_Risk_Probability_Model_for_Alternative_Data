"""
Model training script for credit risk probability model.
Implements Logistic Regression (with WoE) and Gradient Boosting models.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from src.data_processing import CreditRiskDataProcessor


class CreditRiskModelTrainer:
    """
    Trainer for credit risk models.
    Supports Logistic Regression and Gradient Boosting models.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize trainer.
        
        Args:
            model_dir: Directory to save models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.processor = CreditRiskDataProcessor()
        self.scaler = RobustScaler()  # Use RobustScaler for outlier robustness
        self.models = {}
        self.metrics = {}
        
    def train(
        self, 
        data_path: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train credit risk models.
        
        Args:
            data_path: Path to training data
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with training results and metrics
        """
        print("="*80)
        print("TRAINING CREDIT RISK MODELS")
        print("="*80)
        
        # Process data
        print("\n[1/5] Processing data...")
        X, y = self.processor.process_data(data_path, is_training=True)
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        print(f"  Class imbalance: {(y == 1).sum()} high risk, {(y == 0).sum()} low risk")
        
        # Split data
        print("\n[2/5] Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"  Train set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("\n[3/5] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("\n[4/5] Training models...")
        results = {}
        
        # 1. Logistic Regression (interpretable model)
        print("\n  Training Logistic Regression...")
        lr_model = self._train_logistic_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        self.models['logistic_regression'] = lr_model
        results['logistic_regression'] = self.metrics['logistic_regression']
        
        # 2. XGBoost (high performance model)
        print("\n  Training XGBoost...")
        xgb_model = self._train_xgboost(
            X_train, y_train, X_test, y_test
        )
        self.models['xgboost'] = xgb_model
        results['xgboost'] = self.metrics['xgboost']
        
        # 3. LightGBM (alternative high performance model)
        if LIGHTGBM_AVAILABLE:
            print("\n  Training LightGBM...")
            lgb_model = self._train_lightgbm(
                X_train, y_train, X_test, y_test
            )
            self.models['lightgbm'] = lgb_model
            results['lightgbm'] = self.metrics['lightgbm']
        else:
            print("\n  Skipping LightGBM (not installed)")
        
        # Save models
        print("\n[5/5] Saving models...")
        self._save_models()
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def _train_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> LogisticRegression:
        """Train Logistic Regression model."""
        # Use class_weight to handle imbalance
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.metrics['logistic_regression'] = metrics
        
        return model
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle imbalance
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.metrics['xgboost'] = metrics
        
        return model
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):
        """Train LightGBM model."""
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.metrics['lightgbm'] = metrics
        
        return model
    
    def _calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
    
    def _save_models(self):
        """Save trained models and preprocessing objects."""
        # Save models
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved {model_name} to {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Saved scaler to {scaler_path}")
        
        # Save processor (with encoders)
        processor_path = self.model_dir / "processor.pkl"
        with open(processor_path, 'wb') as f:
            pickle.dump(self.processor, f)
        print(f"  Saved processor to {processor_path}")
        
        # Save metrics
        metrics_path = self.model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_path}")
        
        # Save feature names
        feature_names_path = self.model_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.processor.get_feature_names(), f, indent=2)
        print(f"  Saved feature names to {feature_names_path}")


if __name__ == "__main__":
    # Train models
    trainer = CreditRiskModelTrainer()
    results = trainer.train("data/raw/data.csv")
    
    print("\nTraining completed successfully!")
