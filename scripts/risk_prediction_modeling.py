"""
Risk Prediction Modeling with Feature Selection and Class Imbalance Handling
Trains XGBoost and Logistic Regression models with both class weights and SMOTE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

# Class Imbalance Handling
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imbalanced-learn not installed. SMOTE will be skipped.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

# Add config directory to path
sys.path.append(str(Path(__file__).parent.parent / 'config'))
from modeling_config import *

print("=" * 80)
print("RISK PREDICTION MODELING")
print("=" * 80)
print("Features: VIF-selected | Models: XGBoost + Logistic Regression")
print("Class Imbalance: Class Weights + SMOTE")

# Set up paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'
MODELS_DIR = Path(__file__).parent.parent / 'models'
CONFIG_DIR = Path(__file__).parent.parent / 'config'
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. Load Data and VIF-Selected Features
# ============================================================================

print("\n[1] Loading data and VIF-selected features...")
df = pd.read_excel(PROCESSED_DIR / 'data_cleaned.xlsx')
print(f"    Dataset shape: {df.shape}")

# Load VIF-selected features
try:
    from selected_features_vif import SELECTED_FEATURES_VIF
    selected_features = SELECTED_FEATURES_VIF.copy()
    print(f"    Loaded {len(selected_features)} VIF-selected features")
except ImportError:
    print("    WARNING: VIF-selected features not found. Running feature selection first...")
    print("    Please run: python scripts/feature_selection_vif.py")
    # Fallback: use all pre-surgery features
    exclude_features = [
        'Complication Post Surgery', 'Complication During Surgery',
        'Cardiac Complication', 'Pulmonary complication', 'Renal complication',
        'Neurological complication', 'Death post surgery during hospitalization',
        'BUN day 1 post surgery', 'Creatinine_D1', 'Na day 1 post surgery',
        'HB day 1 post surgery', 'Platelet day 1 post surgery',
    ]
    selected_features = [col for col in df.columns if col not in exclude_features]
    print(f"    Using fallback: {len(selected_features)} features")

# Separate features and target
y = df[TARGET_VARIABLE].copy()

# Convert target to binary if needed
if y.dtype == 'object':
    y = y.map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 1: 1, 0: 0})

# Select features
X = df[selected_features].copy()

print(f"    Features: {X.shape[1]}")
print(f"    Target distribution:\n{y.value_counts()}")

# ============================================================================
# 2. Data Preprocessing
# ============================================================================

print("\n[2] Preprocessing data...")

# Handle missing values
print("    Handling missing values...")
numerical_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
        X[col].fillna(mode_val, inplace=True)

# Encode categorical variables
print("    Encoding categorical variables...")
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    if col in X_encoded.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

# Handle datetime columns
datetime_cols = X_encoded.select_dtypes(include=['datetime64']).columns.tolist()
if datetime_cols:
    for col in datetime_cols:
        try:
            X_encoded[col] = (X_encoded[col] - X_encoded[col].min()).dt.days
        except:
            X_encoded = X_encoded.drop(columns=[col])

# Remove zero-variance columns
numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
zero_var_cols = [col for col in numeric_cols if X_encoded[col].var() == 0 or X_encoded[col].nunique() <= 1]
if zero_var_cols:
    X_encoded = X_encoded.drop(columns=zero_var_cols)
    print(f"    Removed {len(zero_var_cols)} zero-variance columns")

# Ensure all columns are numeric
non_numeric_cols = X_encoded.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    for col in non_numeric_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        if col not in label_encoders:
            label_encoders[col] = le

print(f"    Final feature shape: {X_encoded.shape}")

# Train-test split
print("\n[3] Creating train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
    stratify=y if STRATIFY else None
)
print(f"    Train: {X_train.shape[0]} samples")
print(f"    Test: {X_test.shape[0]} samples")
print(f"    Train target distribution:\n{y_train.value_counts()}")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. Class Imbalance Handling
# ============================================================================

print("\n[4] Preparing class imbalance handling...")

# Calculate class weights for XGBoost
n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()
scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

print(f"    Class distribution: {n_negative} negative, {n_positive} positive")
print(f"    Scale pos weight: {scale_pos_weight:.2f}")

# Prepare SMOTE if available
if HAS_SMOTE and USE_SMOTE:
    print("    SMOTE available: Will create oversampled training sets")
    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=SMOTE_RANDOM_STATE)
else:
    print("    SMOTE not available: Will use class weights only")
    USE_SMOTE = False

# ============================================================================
# 4. Model Training
# ============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

models = {}
results = {}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ============================================================================
# Model 1: Logistic Regression with Class Weights
# ============================================================================

print("\n[Model 1A] Logistic Regression with Class Weights...")
print("  Hyperparameter tuning...")

lr_model_cw = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=2000)
lr_grid_cw = GridSearchCV(
    lr_model_cw, LR_PARAM_GRID, cv=cv, scoring=PRIMARY_METRIC,
    n_jobs=-1, verbose=0
)
lr_grid_cw.fit(X_train_scaled, y_train)

print(f"  Best parameters: {lr_grid_cw.best_params_}")
print(f"  Best CV score: {lr_grid_cw.best_score_:.4f}")

lr_final_cw = lr_grid_cw.best_estimator_
y_pred_lr_cw = lr_final_cw.predict(X_test_scaled)
y_pred_proba_lr_cw = lr_final_cw.predict_proba(X_test_scaled)[:, 1]

models['Logistic Regression (Class Weights)'] = lr_final_cw
results['Logistic Regression (Class Weights)'] = {
    'predictions': y_pred_lr_cw,
    'probabilities': y_pred_proba_lr_cw,
    'best_params': lr_grid_cw.best_params_,
    'cv_score': lr_grid_cw.best_score_,
    'scaler': scaler,
    'encoders': label_encoders
}

# ============================================================================
# Model 2: Logistic Regression with SMOTE
# ============================================================================

if USE_SMOTE:
    print("\n[Model 1B] Logistic Regression with SMOTE...")
    print("  Applying SMOTE...")
    
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    print(f"  After SMOTE: {X_train_smote.shape[0]} samples")
    print(f"  SMOTE target distribution:\n{pd.Series(y_train_smote).value_counts()}")
    
    print("  Hyperparameter tuning...")
    lr_model_smote = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)
    lr_grid_smote = GridSearchCV(
        lr_model_smote, LR_PARAM_GRID, cv=cv, scoring=PRIMARY_METRIC,
        n_jobs=-1, verbose=0
    )
    lr_grid_smote.fit(X_train_smote, y_train_smote)
    
    print(f"  Best parameters: {lr_grid_smote.best_params_}")
    print(f"  Best CV score: {lr_grid_smote.best_score_:.4f}")
    
    lr_final_smote = lr_grid_smote.best_estimator_
    y_pred_lr_smote = lr_final_smote.predict(X_test_scaled)
    y_pred_proba_lr_smote = lr_final_smote.predict_proba(X_test_scaled)[:, 1]
    
    models['Logistic Regression (SMOTE)'] = lr_final_smote
    results['Logistic Regression (SMOTE)'] = {
        'predictions': y_pred_lr_smote,
        'probabilities': y_pred_proba_lr_smote,
        'best_params': lr_grid_smote.best_params_,
        'cv_score': lr_grid_smote.best_score_,
        'scaler': scaler,
        'encoders': label_encoders
    }

# ============================================================================
# Model 3: XGBoost with Class Weights
# ============================================================================

if HAS_XGBOOST:
    print("\n[Model 2A] XGBoost with Class Weights...")
    print("  Hyperparameter tuning...")
    
    xgb_param_grid_cw = XGB_PARAM_GRID.copy()
    xgb_param_grid_cw['scale_pos_weight'] = [scale_pos_weight]
    
    xgb_model_cw = xgb.XGBClassifier(
        random_state=RANDOM_STATE, 
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_grid_cw = RandomizedSearchCV(
        xgb_model_cw, xgb_param_grid_cw, cv=cv, scoring=PRIMARY_METRIC,
        n_jobs=-1, verbose=0, n_iter=XGB_N_ITER, random_state=RANDOM_STATE
    )
    xgb_grid_cw.fit(X_train, y_train)
    
    print(f"  Best parameters: {xgb_grid_cw.best_params_}")
    print(f"  Best CV score: {xgb_grid_cw.best_score_:.4f}")
    
    xgb_final_cw = xgb_grid_cw.best_estimator_
    y_pred_xgb_cw = xgb_final_cw.predict(X_test)
    y_pred_proba_xgb_cw = xgb_final_cw.predict_proba(X_test)[:, 1]
    
    models['XGBoost (Class Weights)'] = xgb_final_cw
    results['XGBoost (Class Weights)'] = {
        'predictions': y_pred_xgb_cw,
        'probabilities': y_pred_proba_xgb_cw,
        'best_params': xgb_grid_cw.best_params_,
        'cv_score': xgb_grid_cw.best_score_,
        'scaler': None,  # XGBoost doesn't need scaling
        'encoders': label_encoders
    }

# ============================================================================
# Model 4: XGBoost with SMOTE
# ============================================================================

if HAS_XGBOOST and USE_SMOTE:
    print("\n[Model 2B] XGBoost with SMOTE...")
    print("  Applying SMOTE...")
    
    X_train_smote_xgb, y_train_smote_xgb = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {X_train_smote_xgb.shape[0]} samples")
    print(f"  SMOTE target distribution:\n{pd.Series(y_train_smote_xgb).value_counts()}")
    
    print("  Hyperparameter tuning...")
    xgb_model_smote = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_grid_smote = RandomizedSearchCV(
        xgb_model_smote, XGB_PARAM_GRID, cv=cv, scoring=PRIMARY_METRIC,
        n_jobs=-1, verbose=0, n_iter=XGB_N_ITER, random_state=RANDOM_STATE
    )
    xgb_grid_smote.fit(X_train_smote_xgb, y_train_smote_xgb)
    
    print(f"  Best parameters: {xgb_grid_smote.best_params_}")
    print(f"  Best CV score: {xgb_grid_smote.best_score_:.4f}")
    
    xgb_final_smote = xgb_grid_smote.best_estimator_
    y_pred_xgb_smote = xgb_final_smote.predict(X_test)
    y_pred_proba_xgb_smote = xgb_final_smote.predict_proba(X_test)[:, 1]
    
    models['XGBoost (SMOTE)'] = xgb_final_smote
    results['XGBoost (SMOTE)'] = {
        'predictions': y_pred_xgb_smote,
        'probabilities': y_pred_proba_xgb_smote,
        'best_params': xgb_grid_smote.best_params_,
        'cv_score': xgb_grid_smote.best_score_,
        'scaler': None,
        'encoders': label_encoders
    }

# ============================================================================
# 5. Model Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

evaluation_results = []

for model_name, result in results.items():
    y_pred = result['predictions']
    y_pred_proba = result['probabilities']
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    evaluation_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'CV Score': result['cv_score']
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  CV Score:  {result['cv_score']:.4f}")

# Create comparison dataframe
comparison_df = pd.DataFrame(evaluation_results)
comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)

print("\n" + "=" * 80)
print("MODEL COMPARISON (sorted by ROC-AUC)")
print("=" * 80)
print(comparison_df.to_string(index=False))

# ============================================================================
# 6. Visualizations
# ============================================================================

print("\n[6] Creating visualizations...")

# ROC Curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    y_pred_proba = result['probabilities']
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Risk Prediction Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'risk_model_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: reports/risk_model_roc_curves.png")

# Precision-Recall Curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    y_pred_proba = result['probabilities']
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    plt.plot(recall_curve, precision_curve, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Risk Prediction Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'risk_model_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: reports/risk_model_pr_curves.png")

# Calibration Curves
plt.figure(figsize=(10, 8))
for model_name, result in results.items():
    y_pred_proba = result['probabilities']
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=10
    )
    plt.plot(mean_predicted_value, fraction_of_positives, 
             label=f'{model_name}', linewidth=2, marker='o', markersize=4)

plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title('Calibration Curves - Risk Prediction Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'risk_model_calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: reports/risk_model_calibration.png")

# Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(comparison_df))
width = 0.12

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))

for i, metric in enumerate(metrics):
    offset = (i - len(metrics)/2) * width
    ax.bar(x + offset, comparison_df[metric], width, label=metric, color=colors[i])

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison - Risk Prediction', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=9, ncol=3)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'risk_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: reports/risk_model_comparison.png")

# Feature Importance (for XGBoost models)
xgb_models = {name: model for name, model in models.items() if 'XGBoost' in name}
if xgb_models:
    fig, axes = plt.subplots(1, len(xgb_models), figsize=(6*len(xgb_models), 8))
    if len(xgb_models) == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(xgb_models.items()):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        axes[idx].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[idx].set_yticks(range(len(feature_importance)))
        axes[idx].set_yticklabels(feature_importance['feature'], fontsize=9)
        axes[idx].set_xlabel('Importance', fontsize=11)
        axes[idx].set_title(f'Top 20 Features - {model_name}', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'risk_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: reports/risk_feature_importance.png")

# ============================================================================
# 7. Save Results
# ============================================================================

print("\n[7] Saving results...")

# Save model comparison
comparison_df.to_excel(REPORTS_DIR / 'risk_model_comparison.xlsx', index=False)
print("  Saved: reports/risk_model_comparison.xlsx")

# Save detailed evaluation report
with open(REPORTS_DIR / 'risk_model_evaluation.txt', 'w', encoding='utf-8') as f:
    f.write("RISK PREDICTION MODEL EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Features: {len(selected_features)} VIF-selected features\n")
    f.write(f"Test set size: {len(y_test)} samples\n\n")
    
    f.write("MODEL COMPARISON:\n")
    f.write("-" * 80 + "\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("DETAILED CLASSIFICATION REPORTS:\n")
    f.write("=" * 80 + "\n\n")
    
    for model_name, result in results.items():
        f.write(f"\n{model_name}\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(y_test, result['predictions']))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, result['predictions'])))
        f.write("\n\nBest Parameters:\n")
        for param, value in result['best_params'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

print("  Saved: reports/risk_model_evaluation.txt")

# ============================================================================
# 8. Select and Save Best Model
# ============================================================================

print("\n[8] Selecting best model...")

best_model_name = comparison_df.iloc[0]['Model']
best_model = models[best_model_name]
best_result = results[best_model_name]

print(f"  Best model: {best_model_name}")
print(f"  ROC-AUC: {comparison_df.iloc[0]['ROC-AUC']:.4f}")
print(f"  PR-AUC: {comparison_df.iloc[0]['PR-AUC']:.4f}")

# Save best model and components
joblib.dump(best_model, MODELS_DIR / 'best_risk_model.pkl')
print("  Saved: models/best_risk_model.pkl")

if best_result['scaler'] is not None:
    joblib.dump(best_result['scaler'], MODELS_DIR / 'risk_scaler.pkl')
    print("  Saved: models/risk_scaler.pkl")
else:
    # Create dummy scaler for consistency
    joblib.dump(None, MODELS_DIR / 'risk_scaler.pkl')

joblib.dump(best_result['encoders'], MODELS_DIR / 'risk_encoders.pkl')
print("  Saved: models/risk_encoders.pkl")

joblib.dump(list(X_train.columns), MODELS_DIR / 'risk_feature_columns.pkl')
print("  Saved: models/risk_feature_columns.pkl")

joblib.dump(selected_features, MODELS_DIR / 'risk_selected_features.pkl')
print("  Saved: models/risk_selected_features.pkl")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'roc_auc': float(comparison_df.iloc[0]['ROC-AUC']),
    'pr_auc': float(comparison_df.iloc[0]['PR-AUC']),
    'best_params': best_result['best_params'],
    'cv_score': float(best_result['cv_score']),
    'n_features': len(selected_features),
    'test_size': len(y_test),
    'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(model_metadata, MODELS_DIR / 'risk_model_metadata.pkl')
print("  Saved: models/risk_model_metadata.pkl")

print("\n" + "=" * 80)
print("RISK PREDICTION MODELING COMPLETE!")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"ROC-AUC Score: {comparison_df.iloc[0]['ROC-AUC']:.4f}")
print(f"PR-AUC Score: {comparison_df.iloc[0]['PR-AUC']:.4f}")
print(f"\nAll results saved to: reports/")
print(f"Best model saved to: models/")
print(f"\nTo use the model for predictions:")
print(f"  - Load model: joblib.load('models/best_risk_model.pkl')")
print(f"  - Get risk percentage: probability * 100")
