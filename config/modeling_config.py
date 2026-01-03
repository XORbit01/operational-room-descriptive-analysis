"""
Modeling Configuration for Risk Prediction
Configuration parameters for feature selection and model training
"""

# VIF Configuration
VIF_THRESHOLD = 10.0  # Remove features with VIF > threshold
VIF_MAX_ITERATIONS = 10  # Max iterations for iterative VIF removal

# Class Imbalance Handling
USE_SMOTE = True  # Use SMOTE for oversampling
USE_CLASS_WEIGHTS = True  # Use class weights
SMOTE_K_NEIGHBORS = 5  # Number of neighbors for SMOTE
SMOTE_RANDOM_STATE = 42  # Random state for SMOTE

# Model Selection
PRIMARY_METRIC = 'roc_auc'  # Primary metric for model selection
CV_FOLDS = 5  # Number of cross-validation folds
RANDOM_STATE = 42  # Random state for reproducibility

# Train-Test Split
TEST_SIZE = 0.2  # 20% for testing
STRATIFY = True  # Use stratified split

# Logistic Regression Hyperparameters
LR_PARAM_GRID = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # For elasticnet
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000]
}

# XGBoost Hyperparameters
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# XGBoost RandomizedSearchCV
XGB_N_ITER = 50  # Number of iterations for RandomizedSearchCV

# Target Variable
TARGET_VARIABLE = 'Complication Post Surgery'

# Evaluation Metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'average_precision'  # PR-AUC
]


