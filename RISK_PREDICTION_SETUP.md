# Risk Prediction Model Setup Guide

## Overview

This guide explains how to use the new risk prediction modeling pipeline that:
- Uses VIF-based feature selection to remove redundant features
- Trains XGBoost and Logistic Regression models
- Handles class imbalance with both class weights and SMOTE
- Outputs risk percentages (0-100%) for complication prediction

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `imbalanced-learn>=0.11.0` (for SMOTE)
- `xgboost>=2.0.0` (for XGBoost)
- All other required packages

### Step 2: Run Feature Selection

```bash
python scripts/feature_selection_vif.py
```

This will:
- Load cleaned data
- Select pre-surgery features only
- Calculate VIF for all features
- Remove features with VIF > 10
- Save selected features to `config/selected_features_vif.py`
- Generate report: `reports/feature_selection_vif_report.txt`

**Output:**
- `config/selected_features_vif.py` - Selected features list
- `config/selected_features_vif.pkl` - Pickle file for easy loading
- `reports/feature_selection_vif_report.txt` - Detailed report

### Step 3: Train Risk Prediction Models

```bash
python scripts/risk_prediction_modeling.py
```

This will:
- Load VIF-selected features
- Train 4 models:
  1. Logistic Regression (Class Weights)
  2. Logistic Regression (SMOTE)
  3. XGBoost (Class Weights)
  4. XGBoost (SMOTE)
- Evaluate all models
- Select best model based on ROC-AUC
- Save best model and components

**Output:**
- `models/best_risk_model.pkl` - Best trained model
- `models/risk_scaler.pkl` - Feature scaler (if needed)
- `models/risk_encoders.pkl` - Label encoders
- `models/risk_feature_columns.pkl` - Feature column order
- `models/risk_selected_features.pkl` - Selected features
- `models/risk_model_metadata.pkl` - Model metadata
- `reports/risk_model_comparison.xlsx` - Model comparison table
- `reports/risk_model_evaluation.txt` - Detailed evaluation
- `reports/risk_model_roc_curves.png` - ROC curves
- `reports/risk_model_pr_curves.png` - Precision-Recall curves
- `reports/risk_model_calibration.png` - Calibration curves
- `reports/risk_model_comparison.png` - Performance comparison
- `reports/risk_feature_importance.png` - Feature importance (XGBoost)

## Using the Model in Dashboard

The dashboard has been updated with new functions in `dashboard/utils/model_loader.py`:

### Load Risk Model

```python
from dashboard.utils.model_loader import load_risk_model, predict_risk, get_risk_percentage

model, scaler, encoders, feature_columns, metadata, error = load_risk_model()
```

### Make Predictions

```python
# Prepare input data as dictionary
input_data = {
    'Age': 65,
    'Gender': 'male',
    'BMI': 28.5,
    'Hypertension': 'yes',
    'DiabetesMellitus': 'no',
    # ... other features
}

# Load template data (cleaned dataset)
from dashboard.utils.data_loader import load_data
df_template = load_data()

# Predict risk
result = predict_risk(input_data, df_template)

if result['error']:
    print(f"Error: {result['error']}")
else:
    print(f"Risk Percentage: {result['risk_percentage']}%")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Prediction: {'Complication' if result['prediction'] == 1 else 'No Complication'}")
```

### Convert Probability to Risk Percentage

```python
probability = 0.35  # Model output (0-1 range)
risk_percentage = get_risk_percentage(probability)  # Returns 35.0
```

## Configuration

Modeling parameters can be adjusted in `config/modeling_config.py`:

- `VIF_THRESHOLD = 10.0` - VIF threshold for feature removal
- `USE_SMOTE = True` - Enable SMOTE oversampling
- `USE_CLASS_WEIGHTS = True` - Enable class weights
- `PRIMARY_METRIC = 'roc_auc'` - Metric for model selection
- `CV_FOLDS = 5` - Cross-validation folds

## Model Selection

The best model is selected based on ROC-AUC score. All models are evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (Average Precision)

## Feature Selection Details

### VIF Analysis
- Removes features with VIF > 10 (indicating multicollinearity)
- Iteratively removes highest VIF features until all < 10
- Preserves clinically important features

### Special Handling
- **Weight vs BMI**: If both have high VIF, keeps the one with lower VIF
- **Data Leakage**: Removes `complication_count` (correlates 0.93 with target)
- **Post-Surgery Features**: Automatically excluded

## Class Imbalance Handling

### Class Weights
- **Logistic Regression**: Uses `class_weight='balanced'`
- **XGBoost**: Uses `scale_pos_weight = n_negative / n_positive`

### SMOTE
- Synthetic Minority Oversampling Technique
- Creates synthetic samples for minority class
- Applied only to training data
- Parameters: `k_neighbors=5`

## Expected Results

After running the pipeline:
- Feature reduction: ~20-30% (removes redundant features)
- Model performance: ROC-AUC > 0.70 (baseline to beat)
- Best model: Selected from 4 trained models
- Risk percentages: Calibrated probabilities (0-100%)

## Troubleshooting

### Error: "VIF-selected features not found"
**Solution:** Run `python scripts/feature_selection_vif.py` first

### Error: "imbalanced-learn not installed"
**Solution:** `pip install imbalanced-learn>=0.11.0`

### Error: "XGBoost not installed"
**Solution:** `pip install xgboost>=2.0.0`

### Model performance is low
**Possible causes:**
- Small dataset (522 samples) - consider collecting more data
- High class imbalance - try adjusting SMOTE parameters
- Feature quality - review feature selection report

## Next Steps

1. Review `reports/feature_selection_vif_report.txt` to see which features were removed
2. Check `reports/risk_model_comparison.xlsx` to compare model performance
3. Review `reports/risk_model_evaluation.txt` for detailed metrics
4. Use the best model in the dashboard for risk predictions

## Files Created

### Scripts
- `scripts/feature_selection_vif.py` - VIF-based feature selection
- `scripts/risk_prediction_modeling.py` - Model training with imbalance handling

### Configuration
- `config/modeling_config.py` - Modeling parameters
- `config/selected_features_vif.py` - Auto-generated selected features

### Updated Files
- `requirements.txt` - Added imbalanced-learn and xgboost
- `dashboard/utils/model_loader.py` - Added risk prediction functions


