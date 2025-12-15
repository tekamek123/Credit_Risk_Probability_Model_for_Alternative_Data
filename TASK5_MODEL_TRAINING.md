# Task 5: Model Training and Tracking - Implementation Summary

## Overview
Implemented comprehensive model training pipeline with MLflow experiment tracking, multiple models, hyperparameter tuning, and comprehensive evaluation metrics.

## Requirements Met

### ✅ 1. Setup
- Added `mlflow>=2.8.0` to `requirements.txt`
- `pytest` was already in requirements.txt
- MLflow import is optional (graceful degradation if not installed)

### ✅ 2. Data Preparation
- **Train/Test Split**: Implemented with `train_test_split` from sklearn
- **Reproducibility**: `random_state=42` parameter ensures consistent splits
- **Stratified Split**: Maintains class distribution in train/test sets
- **Test Size**: Configurable (default: 0.2 = 20% test, 80% train)

### ✅ 3. Model Selection and Training
Implemented **5 models**:

1. **Logistic Regression** (interpretable, Basel II compliant)
   - Uses `RobustScaler` for outlier robustness
   - Class weighting for imbalanced data

2. **Decision Tree** (simple, interpretable)
   - Configurable max_depth, min_samples_split, min_samples_leaf

3. **Random Forest** (ensemble, robust)
   - Multiple estimators with parallel processing

4. **XGBoost** (gradient boosting, high performance)
   - Scale_pos_weight for class imbalance
   - Early stopping support

5. **LightGBM** (gradient boosting, fast)
   - Optional (if library installed)
   - Scale_pos_weight for class imbalance

### ✅ 4. Hyperparameter Tuning
Implemented both methods:

- **Grid Search** (`GridSearchCV`):
  - Exhaustive search over parameter grid
  - 5-fold cross-validation
  - ROC-AUC scoring

- **Random Search** (`RandomizedSearchCV`):
  - Random sampling of parameter combinations
  - Faster for large parameter spaces
  - 10 iterations, 5-fold CV

**Hyperparameter Grids:**
- **Logistic Regression**: C, solver, class_weight
- **Decision Tree**: max_depth, min_samples_split, min_samples_leaf, class_weight
- **Random Forest**: n_estimators, max_depth, min_samples_split, class_weight
- **XGBoost**: n_estimators, max_depth, learning_rate, subsample
- **LightGBM**: n_estimators, max_depth, learning_rate, subsample

### ✅ 5. Experiment Tracking with MLflow
Comprehensive MLflow integration:

- **Parameters Logged**:
  - Model hyperparameters
  - Model type
  - Number of features
  - Training/test set sizes

- **Metrics Logged**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

- **Artifacts Logged**:
  - Trained models (sklearn, xgboost, lightgbm formats)
  - Model signatures for inference

- **Model Registry**:
  - Best model automatically registered
  - Model versioning and tracking
  - Production-ready model management

- **MLflow UI**:
  - Compare model runs
  - Visualize metrics
  - Track experiments

### ✅ 6. Model Evaluation
Comprehensive metrics implemented:

1. **Accuracy**: Ratio of correctly predicted observations
2. **Precision**: Ratio of correctly predicted positives to total predicted positives
3. **Recall (Sensitivity)**: Ratio of correctly predicted positives to all actual positives
4. **F1 Score**: Weighted average of Precision and Recall
5. **ROC-AUC**: Area Under the ROC Curve (primary metric for model selection)

All metrics calculated using sklearn functions with proper zero-division handling.

### ✅ 7. Unit Tests
Implemented **4+ unit tests** for helper functions:

1. `test_calculate_metrics`: Tests metric calculation with real data
2. `test_calculate_metrics_with_perfect_predictions`: Edge case testing
3. `test_identify_best_model`: Tests best model selection logic
4. `test_data_splitting_reproducibility`: Tests reproducible data splitting
5. `test_feature_engineering_returns_expected_columns`: Tests feature engineering output
6. `test_feature_engineering_no_missing_values`: Tests data quality

## Implementation Details

### MLflow Context Manager
Created `MLflowContext` class for optional MLflow support:
- Gracefully handles MLflow not being installed
- Maintains code functionality without MLflow
- Easy to enable/disable tracking

### Best Model Selection
- Automatically identifies best model based on ROC-AUC score
- Registers best model in MLflow Model Registry
- Saves all models locally for comparison

### Model Persistence
- All models saved as pickle files
- Scaler and processor saved for inference
- Metrics saved as JSON for analysis
- Feature names saved for consistency

## Usage Example

```python
from src.train import CreditRiskModelTrainer

# Initialize trainer
trainer = CreditRiskModelTrainer(
    model_dir="models",
    experiment_name="credit_risk_modeling"
)

# Train models with hyperparameter tuning
results = trainer.train(
    "data/raw/data.csv",
    test_size=0.2,
    random_state=42,
    use_hyperparameter_tuning=True,
    tuning_method='random'  # or 'grid'
)

# View results
print(f"Best model: {trainer.best_model_name}")
print(f"Best ROC-AUC: {trainer.best_score}")

# View MLflow UI: mlflow ui
```

## Test Results
- ✅ 6+ unit tests, all passing
- ✅ Tests cover helper functions, feature engineering, and training pipeline
- ✅ Reproducibility verified
- ✅ Edge cases handled

## Files Modified/Created
- `src/train.py` - Complete rewrite with MLflow tracking, multiple models, hyperparameter tuning
- `requirements.txt` - Added mlflow
- `tests/test_train.py` - Comprehensive unit tests
- `TASK5_MODEL_TRAINING.md` - This documentation

## MLflow UI Access
After training, view experiments:
```bash
mlflow ui
```
Then open `http://localhost:5000` in your browser to:
- Compare model runs
- View metrics and parameters
- Download models
- Access Model Registry

## Next Steps
The trained models are ready for:
- Model evaluation and comparison
- Production deployment
- API integration
- Model monitoring and retraining

