"""
Model training script for credit risk probability model with MLflow tracking.
Implements multiple models, hyperparameter tuning, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

from src.data_processing import CreditRiskDataProcessor


class MLflowContext:
    """Context manager for optional MLflow runs."""
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.context = None
    
    def __enter__(self):
        if MLFLOW_AVAILABLE:
            self.context = mlflow.start_run(run_name=self.run_name)
            self.context.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            self.context.__exit__(exc_type, exc_val, exc_tb)


class CreditRiskModelTrainer:
    """
    Trainer for credit risk models with MLflow tracking.
    Supports multiple models, hyperparameter tuning, and experiment tracking.
    """
    
    def __init__(
        self, 
        model_dir: str = "models",
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "credit_risk_modeling"
    ):
        """
        Initialize trainer.
        
        Args:
            model_dir: Directory to save models
            mlflow_tracking_uri: MLflow tracking URI (default: local file store)
            experiment_name: MLflow experiment name
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.processor = CreditRiskDataProcessor()
        self.scaler = RobustScaler()
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
        # Setup MLflow
        if not MLFLOW_AVAILABLE:
            warnings.warn("MLflow not available. Install with: pip install mlflow")
        else:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
        
    def train(
        self, 
        data_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        use_hyperparameter_tuning: bool = True,
        tuning_method: str = 'grid'  # 'grid' or 'random'
    ) -> Dict[str, Any]:
        """
        Train credit risk models with MLflow tracking.
        
        Args:
            data_path: Path to training data
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            use_hyperparameter_tuning: Whether to perform hyperparameter tuning
            tuning_method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
            
        Returns:
            Dictionary with training results and metrics
        """
        print("="*80)
        print("TRAINING CREDIT RISK MODELS WITH MLFLOW TRACKING")
        print("="*80)
        
        # Process data
        print("\n[1/6] Processing data...")
        X, y = self.processor.process_data(data_path, is_training=True, scaling_method='robust')
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        print(f"  Class imbalance: {(y == 1).sum()} high risk, {(y == 0).sum()} low risk")
        
        # Split data with reproducibility
        print(f"\n[2/6] Splitting data (test_size={test_size}, random_state={random_state})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"  Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("\n[3/6] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Train models with MLflow tracking
        print("\n[4/6] Training models with MLflow tracking...")
        results = {}
        
        # 1. Logistic Regression
        print("\n  [1/5] Training Logistic Regression...")
        lr_result = self._train_logistic_regression(
            X_train_scaled, y_train, X_test_scaled, y_test,
            use_hyperparameter_tuning, tuning_method
        )
        results['logistic_regression'] = lr_result
        
        # 2. Decision Tree
        print("\n  [2/5] Training Decision Tree...")
        dt_result = self._train_decision_tree(
            X_train, y_train, X_test, y_test,
            use_hyperparameter_tuning, tuning_method
        )
        results['decision_tree'] = dt_result
        
        # 3. Random Forest
        print("\n  [3/5] Training Random Forest...")
        rf_result = self._train_random_forest(
            X_train, y_train, X_test, y_test,
            use_hyperparameter_tuning, tuning_method
        )
        results['random_forest'] = rf_result
        
        # 4. XGBoost
        print("\n  [4/5] Training XGBoost...")
        xgb_result = self._train_xgboost(
            X_train, y_train, X_test, y_test,
            use_hyperparameter_tuning, tuning_method
        )
        results['xgboost'] = xgb_result
        
        # 5. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            print("\n  [5/5] Training LightGBM...")
            lgb_result = self._train_lightgbm(
                X_train, y_train, X_test, y_test,
                use_hyperparameter_tuning, tuning_method
            )
            results['lightgbm'] = lgb_result
        else:
            print("\n  [5/5] Skipping LightGBM (not installed)")
        
        # Identify best model
        print("\n[5/6] Identifying best model...")
        self._identify_best_model(results)
        
        # Register best model in MLflow Model Registry
        print("\n[6/6] Registering best model in MLflow Model Registry...")
        self._register_best_model(X_test_scaled if self.best_model_name == 'logistic_regression' else X_test, y_test)
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"\nBest Model: {self.best_model_name.upper()}")
        print(f"Best ROC-AUC Score: {self.best_score:.4f}")
        print("\nAll Models Performance:")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return results
    
    def _train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_tuning: bool,
        tuning_method: str
    ) -> Dict[str, Any]:
        """Train Logistic Regression model with MLflow tracking."""
        with MLflowContext("logistic_regression"):
            # Hyperparameter tuning
            if use_tuning:
                param_grid = {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'solver': ['lbfgs', 'liblinear'],
                    'class_weight': ['balanced', None]
                }
                
                base_model = LogisticRegression(max_iter=1000, random_state=42)
                
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        base_model, param_grid, 
                        cv=5, scoring='roc_auc', 
                        n_jobs=-1, verbose=0
                    )
                else:  # random
                    search = RandomizedSearchCV(
                        base_model, param_grid,
                        n_iter=10, cv=5, scoring='roc_auc',
                        n_jobs=-1, random_state=42, verbose=0
                    )
                
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model = LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs'
                )
                model.fit(X_train, y_train)
                best_params = model.get_params()
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.log_param("model_type", "LogisticRegression")
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("n_samples_train", len(X_train))
                mlflow.log_param("n_samples_test", len(X_test))
                
                # Log model
                signature = infer_signature(X_train, y_pred_proba)
                mlflow.sklearn.log_model(model, "model", signature=signature)
            
            # Save model
            self.models['logistic_regression'] = model
            self.metrics['logistic_regression'] = metrics
            
            return metrics
    
    def _train_decision_tree(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_tuning: bool,
        tuning_method: str
    ) -> Dict[str, Any]:
        """Train Decision Tree model with MLflow tracking."""
        with MLflowContext("decision_tree"):
            if use_tuning:
                param_grid = {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', None]
                }
                
                base_model = DecisionTreeClassifier(random_state=42)
                
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        base_model, param_grid,
                        cv=5, scoring='roc_auc',
                        n_jobs=-1, verbose=0
                    )
                else:
                    search = RandomizedSearchCV(
                        base_model, param_grid,
                        n_iter=10, cv=5, scoring='roc_auc',
                        n_jobs=-1, random_state=42, verbose=0
                    )
                
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model = DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=5,
                    class_weight='balanced',
                    random_state=42
                )
                model.fit(X_train, y_train)
                best_params = model.get_params()
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            mlflow.sklearn.log_model(model, "model")
            
            self.models['decision_tree'] = model
            self.metrics['decision_tree'] = metrics
            
            return metrics
    
    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_tuning: bool,
        tuning_method: str
    ) -> Dict[str, Any]:
        """Train Random Forest model with MLflow tracking."""
        with MLflowContext("random_forest"):
            if use_tuning:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced', None]
                }
                
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        base_model, param_grid,
                        cv=5, scoring='roc_auc',
                        n_jobs=-1, verbose=0
                    )
                else:
                    search = RandomizedSearchCV(
                        base_model, param_grid,
                        n_iter=10, cv=5, scoring='roc_auc',
                        n_jobs=-1, random_state=42, verbose=0
                    )
                
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                best_params = model.get_params()
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.sklearn.log_model(model, "model")
            
            self.models['random_forest'] = model
            self.metrics['random_forest'] = metrics
            
            return metrics
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_tuning: bool,
        tuning_method: str
    ) -> Dict[str, Any]:
        """Train XGBoost model with MLflow tracking."""
        with MLflowContext("xgboost"):
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            if use_tuning:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
                
                base_model = xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    eval_metric='logloss'
                )
                
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        base_model, param_grid,
                        cv=5, scoring='roc_auc',
                        n_jobs=-1, verbose=0
                    )
                else:
                    search = RandomizedSearchCV(
                        base_model, param_grid,
                        n_iter=10, cv=5, scoring='roc_auc',
                        n_jobs=-1, random_state=42, verbose=0
                    )
                
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    eval_metric='logloss'
                )
                model.fit(X_train, y_train)
                best_params = model.get_params()
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.xgboost.log_model(model, "model")
            
            self.models['xgboost'] = model
            self.metrics['xgboost'] = metrics
            
            return metrics
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_tuning: bool,
        tuning_method: str
    ) -> Dict[str, Any]:
        """Train LightGBM model with MLflow tracking."""
        with MLflowContext("lightgbm"):
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            if use_tuning:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
                
                base_model = lgb.LGBMClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbose=-1
                )
                
                if tuning_method == 'grid':
                    search = GridSearchCV(
                        base_model, param_grid,
                        cv=5, scoring='roc_auc',
                        n_jobs=-1, verbose=0
                    )
                else:
                    search = RandomizedSearchCV(
                        base_model, param_grid,
                        n_iter=10, cv=5, scoring='roc_auc',
                        n_jobs=-1, random_state=42, verbose=0
                    )
                
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train, y_train)
                best_params = model.get_params()
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log to MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_type", "LGBMClassifier")
            mlflow.lightgbm.log_model(model, "model")
            
            self.models['lightgbm'] = model
            self.metrics['lightgbm'] = metrics
            
            return metrics
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def _identify_best_model(self, results: Dict[str, Dict[str, float]]):
        """Identify best model based on ROC-AUC score."""
        for model_name, metrics in results.items():
            if metrics['roc_auc'] > self.best_score:
                self.best_score = metrics['roc_auc']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"  Best model: {self.best_model_name} (ROC-AUC: {self.best_score:.4f})")
    
    def _register_best_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Register best model in MLflow Model Registry."""
        if self.best_model is None:
            print("  No best model to register.")
            return
        
        if not MLFLOW_AVAILABLE:
            print("  MLflow not available. Skipping model registration.")
            return
        
        # Get the run ID for the best model
        # In a real scenario, you'd query MLflow to find the run
        # For now, we'll create a new run for the registered model
        with MLflowContext(f"{self.best_model_name}_registered"):
            # Log best model info
            mlflow.log_param("best_model_name", self.best_model_name)
            mlflow.log_metric("best_roc_auc", self.best_score)
            
            # Log the model
            if self.best_model_name == 'logistic_regression':
                signature = infer_signature(X_test, self.best_model.predict_proba(X_test)[:, 1])
                mlflow.sklearn.log_model(self.best_model, "model", signature=signature)
            elif self.best_model_name == 'xgboost':
                mlflow.xgboost.log_model(self.best_model, "model")
            elif self.best_model_name == 'lightgbm':
                mlflow.lightgbm.log_model(self.best_model, "model")
            else:
                mlflow.sklearn.log_model(self.best_model, "model")
            
            # Register model
            if MLFLOW_AVAILABLE and mlflow.active_run():
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mlflow.register_model(model_uri, "CreditRiskModel")
                print(f"  Registered model: {model_uri}")
    
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
        
        # Save processor
        processor_path = self.model_dir / "processor.pkl"
        with open(processor_path, 'wb') as f:
            pickle.dump(self.processor, f)
        print(f"  Saved processor to {processor_path}")
        
        # Save metrics
        metrics_path = self.model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_path}")


if __name__ == "__main__":
    # Train models
    trainer = CreditRiskModelTrainer()
    results = trainer.train(
        "data/raw/data.csv",
        test_size=0.2,
        random_state=42,
        use_hyperparameter_tuning=True,
        tuning_method='random'  # Use random search for faster execution
    )
    
    print("\nTraining completed successfully!")
    print(f"\nView MLflow UI: mlflow ui")
    print(f"Best model: {trainer.best_model_name}")
