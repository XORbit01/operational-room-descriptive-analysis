"""
Configuration file for data cleaning and preprocessing parameters
"""

# Missing value thresholds
HIGH_MISSING_THRESHOLD = 0.80  # 80% missing - exclude from main analysis
MEDIUM_MISSING_THRESHOLD = 0.20  # 20% missing - context-specific handling

# KNN Imputation parameters
KNN_K = 5
KNN_FEATURES = ['Age', 'Gender', 'BMI_Category']

# BMI Categories
BMI_UNDERWEIGHT = 18.5
BMI_NORMAL = 25.0
BMI_OVERWEIGHT = 30.0

# Outlier detection
IQR_MULTIPLIER = 1.5  # Standard IQR multiplier for outlier detection

# Medical ranges for validation
AGE_MIN = 0
AGE_MAX = 120
WEIGHT_MIN_KG = 20
WEIGHT_MAX_KG = 300
HEIGHT_MIN_CM = 100
HEIGHT_MAX_CM = 250
BMI_MIN = 10
BMI_MAX = 60

# Columns to drop
COLUMNS_TO_DROP = [
    'Descriptions',
    'Phone number',
    'Systemic Inflammatory Response Syndrome (SISS)',
    'Patient name',
    'ID number'
]

# High missing columns to exclude (>80%)
HIGH_MISSING_COLUMNS = [
    'BUN before Discharge',
    'Urine Output',
    'Blood Loss during surgery',
    'Creatinine before Discharge',
    'Na before Discharge',
    'Tumor Category'
]

# Lab value columns
PRE_SURGERY_LABS = [
    'Pre-BUN',
    'Pre-Creatinine',
    'Pre Na',
    'Pre HB',
    'Pre Platelet'
]

POST_OP_DAY1_LABS = [
    'BUN day 1 post surgery',
    'Creatinine_D1',
    'Na day 1 post surgery',
    'HB day 1 post surgery',
    'Platelet day 1 post surgery'
]

DISCHARGE_LABS = [
    'BUN before Discharge',
    'Creatinine before Discharge',
    'Na before Discharge',
    'HB before discharge',
    'Platelet befor eDischarge'
]

# Post-discharge outcome columns
POST_DISCHARGE_COLUMNS = [
    'Complication post Discharge',
    'ER Visit',
    'Readmission due to OR',
    'Infection or inflammation',
    'Redo surgery',
    'Admission into other hospital',
    'Death post discharge',
    'Type of complication post discharge'
]

# Medical history binary columns
MEDICAL_HISTORY_COLUMNS = [
    'Hypertension',
    'DiabetesMellitus',
    'Dyslipidemia',
    'CAD History',
    'HF',
    'Open heart surgery',
    'AFib-tachycardia',
    'PAD',
    'COPD',
    'CKD',
    'Dialysis',
    'Neurological/ Psychological disease',
    'Gastrointestinal Disease',
    'Endocrine Disease',
    'Cancer',
    'Allergy'
]
