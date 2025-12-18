"""
Main data cleaning pipeline for medical/surgical dataset
Implements all phases of the data cleaning and preprocessing plan
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path

# Add config directory to path
sys.path.append(str(Path(__file__).parent.parent / 'config'))
from config import *

# Add scripts directory to path for data_quality_report
sys.path.append(str(Path(__file__).parent))
from data_quality_report import generate_quality_report

# Import for KNN imputation
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

print("=" * 80)
print("DATA CLEANING AND PREPROCESSING PIPELINE")
print("=" * 80)

# ============================================================================
# PHASE 1: Initial Data Loading and Assessment
# ============================================================================
print("\n[PHASE 1] Loading data and creating assessment report...")

# Load data
DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DATA = DATA_DIR / 'raw' / 'data.xlsx'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'

df = pd.read_excel(RAW_DATA)
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Create backup
df_backup = df.copy()
df_backup.to_excel(DATA_DIR / 'raw' / 'data_backup.xlsx', index=False)
print("Backup saved to: data/raw/data_backup.xlsx")

# Generate quality report
report_text = generate_quality_report(df, str(REPORTS_DIR / 'data_quality_report_initial.txt'))
print("Initial data quality report generated")

# ============================================================================
# PHASE 2: Column Name Standardization
# ============================================================================
print("\n[PHASE 2] Fixing column names...")

# Create column name mapping for typos
column_rename_map = {
    'Platelet befor eDischarge': 'Platelet before Discharge',
    'Number of ransfused packet cells': 'Number of transfused packet cells',
}

# Apply renaming
df = df.rename(columns=column_rename_map)
print(f"Fixed {len(column_rename_map)} column name typos")

# Update DISCHARGE_LABS in config to use corrected name
DISCHARGE_LABS = [
    'BUN before Discharge',
    'Creatinine before Discharge',
    'Na before Discharge',
    'HB before discharge',
    'Platelet before Discharge'  # Updated
]

# ============================================================================
# PHASE 3: Handle Missing Values (Feature-Specific Strategy)
# ============================================================================
print("\n[PHASE 3] Handling missing values with feature-specific strategy...")

# Track imputation log
imputation_log = []

# Group A: Demographics and Anthropometrics (KNN Imputation)
print("\n  [Group A] Demographics - KNN Imputation...")

demo_cols = ['Weight', 'Height', 'BMI']
demo_missing = df[demo_cols].isnull().any(axis=1).sum()
print(f"    Patients with missing demographics: {demo_missing}")

# Prepare for KNN - need to encode categorical features first
df_knn = df.copy()

# Encode Gender for KNN
if 'Gender' in df_knn.columns:
    le_gender = LabelEncoder()
    df_knn['Gender_encoded'] = df_knn['Gender'].map({'male': 0, 'female': 1})

# Encode BMI_Category for KNN (if available)
if 'BMI_Category' in df_knn.columns:
    bmi_cat_map = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
    df_knn['BMI_Category_encoded'] = df_knn['BMI_Category'].map(bmi_cat_map)
    # Fill missing BMI_Category_encoded with median for KNN
    df_knn['BMI_Category_encoded'] = df_knn['BMI_Category_encoded'].fillna(
        df_knn['BMI_Category_encoded'].median()
    )

# Features for KNN similarity
knn_features = ['Age', 'Gender_encoded', 'BMI_Category_encoded']
knn_features = [f for f in knn_features if f in df_knn.columns]

# Prepare data for KNN imputation
knn_data = df_knn[knn_features + demo_cols].copy()

# Fill missing in feature columns with median for KNN
for col in knn_features:
    if knn_data[col].isnull().any():
        knn_data[col] = knn_data[col].fillna(knn_data[col].median())

# Apply KNN imputation
imputer = KNNImputer(n_neighbors=KNN_K)
knn_imputed = imputer.fit_transform(knn_data[demo_cols])

# Update dataframe with imputed values
for i, col in enumerate(demo_cols):
    missing_before = df[col].isnull().sum()
    df[col] = knn_imputed[:, i]
    missing_after = df[col].isnull().sum()
    imputed_count = missing_before - missing_after
    if imputed_count > 0:
        imputation_log.append(f"KNN imputed {imputed_count} values in {col}")
        print(f"    KNN imputed {imputed_count} values in {col}")

# Recalculate BMI from Weight and Height
print("    Recalculating BMI from Weight and Height...")
mask = df['BMI'].isnull() | (df['Weight'].notna() & df['Height'].notna())
df.loc[mask, 'BMI'] = df.loc[mask, 'Weight'] / ((df.loc[mask, 'Height'] / 100) ** 2)

# Derive BMI_Category from BMI
print("    Deriving BMI_Category from BMI...")
def categorize_bmi(bmi):
    if pd.isna(bmi):
        return np.nan
    elif bmi < BMI_UNDERWEIGHT:
        return 'underweight'
    elif bmi < BMI_NORMAL:
        return 'normal'
    elif bmi < BMI_OVERWEIGHT:
        return 'overweight'
    else:
        return 'obese'

df['BMI_Category'] = df['BMI'].apply(categorize_bmi)

# Mode imputation for low missing categorical demographics
print("    Mode imputation for categorical demographics...")
for col in ['Governorate', 'MaritalStatus']:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        if missing_before > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            if mode_val is not None:
                df[col] = df[col].fillna(mode_val)
                imputation_log.append(f"Mode imputed {missing_before} values in {col} (mode: {mode_val})")
                print(f"    Mode imputed {missing_before} values in {col}")

# Group B: Medical History Binary Columns (Mode Imputation)
print("\n  [Group B] Medical History - Mode Imputation...")

for col in MEDICAL_HISTORY_COLUMNS:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        if missing_before > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            if mode_val is not None:
                df[col] = df[col].fillna(mode_val)
                imputation_log.append(f"Mode imputed {missing_before} values in {col} (mode: {mode_val})")
                if missing_before > 5:  # Only print if significant
                    print(f"    Mode imputed {missing_before} values in {col}")

# Group C: Lab Values (Subset Analysis - Complete Cases Only)
print("\n  [Group C] Lab Values - Creating subset datasets...")

# Create lab availability indicators
df['has_pre_labs'] = df[PRE_SURGERY_LABS].notna().all(axis=1).astype(int)
df['has_post_labs'] = df[POST_OP_DAY1_LABS].notna().all(axis=1).astype(int)
df['has_discharge_labs'] = df[DISCHARGE_LABS].notna().all(axis=1).astype(int)

print(f"    Patients with complete pre-surgery labs: {df['has_pre_labs'].sum()}")
print(f"    Patients with complete Day 1 post-op labs: {df['has_post_labs'].sum()}")
print(f"    Patients with complete discharge labs: {df['has_discharge_labs'].sum()}")

# Create subset datasets (will be saved later)
df_pre_labs = df[df['has_pre_labs'] == 1].copy()
df_post_labs = df[df['has_post_labs'] == 1].copy()
df_discharge_labs = df[df['has_discharge_labs'] == 1].copy()

# Group D: Post-Discharge Outcome Variables (Missing Indicator)
print("\n  [Group D] Post-Discharge Outcomes - Creating missing indicator...")

# Create follow-up availability indicator
df['follow_up_available'] = df[POST_DISCHARGE_COLUMNS].notna().any(axis=1).astype(int)
print(f"    Patients with post-discharge follow-up data: {df['follow_up_available'].sum()}")

# Group E: Surgical Details - Context-Specific Handling
print("\n  [Group E] Surgical Details - Context-specific handling...")

# Medium missing - Create "Unknown" category
for col in ['Extubation Post OR', 'ICD10']:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        if missing_before > 0:
            df[col] = df[col].fillna('Unknown')
            imputation_log.append(f"Created 'Unknown' category for {missing_before} values in {col}")
            print(f"    Created 'Unknown' category for {missing_before} values in {col}")

# Low missing - Mode imputation for surgical details
surgical_cols = ['Emergency Status of surgery', 'Anesthesia type', 'Way Of Anesthesia']
for col in surgical_cols:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        if missing_before > 0 and missing_before < len(df) * 0.2:  # <20% missing
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            if mode_val is not None:
                df[col] = df[col].fillna(mode_val)
                imputation_log.append(f"Mode imputed {missing_before} values in {col}")

# Group F: Drop Columns
print("\n  [Group F] Dropping columns not useful for analysis...")
cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"    Dropped {len(cols_to_drop)} columns: {cols_to_drop}")

# Drop high missing columns
cols_to_drop_high = [col for col in HIGH_MISSING_COLUMNS if col in df.columns]
df = df.drop(columns=cols_to_drop_high)
print(f"    Dropped {len(cols_to_drop_high)} high missing columns: {cols_to_drop_high}")

print(f"\n[PHASE 3] Missing value handling complete. Logged {len(imputation_log)} imputation operations.")

# Save imputation log
with open(REPORTS_DIR / 'imputation_log.txt', 'w') as f:
    f.write("IMPUTATION LOG\n")
    f.write("=" * 80 + "\n\n")
    for log_entry in imputation_log:
        f.write(log_entry + "\n")
print("Imputation log saved to: reports/imputation_log.txt")

# ============================================================================
# PHASE 4: Data Type Corrections
# ============================================================================
print("\n[PHASE 4] Fixing data types...")

# Fix BloodGroup - mixed numeric and text
if 'BloodGroup' in df.columns:
    print("  Fixing BloodGroup column (mixed types)...")
    # Convert all to string first
    df['BloodGroup'] = df['BloodGroup'].astype(str)
    # Map numeric codes (if they exist) - keeping original values for now
    # Since we don't know the exact mapping, we'll standardize 'yes' values
    df['BloodGroup'] = df['BloodGroup'].replace('yes', 'Unknown')
    df['BloodGroup'] = df['BloodGroup'].replace('nan', np.nan)
    print(f"    BloodGroup unique values: {df['BloodGroup'].value_counts().head()}")

# Convert numeric to categorical where appropriate
print("  Converting numeric columns to categorical...")
numeric_to_categorical = {
    'Number of transfused PC during surgery': 'category',
    'Number of transfused packet cells': 'category',  # Fixed name
    'Duration in intensive care unit (days)': 'category'
}

for col, dtype in numeric_to_categorical.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)
        print(f"    Converted {col} to {dtype}")

# Standardize categorical values
print("  Standardizing categorical values...")

# Fix MaritalStatus - has numeric 2 mixed with yes/no
if 'MaritalStatus' in df.columns:
    # Map numeric 2 to a category or mode
    if 2 in df['MaritalStatus'].values:
        mode_val = df['MaritalStatus'].mode()[0] if not df['MaritalStatus'].mode().empty else 'yes'
        df['MaritalStatus'] = df['MaritalStatus'].replace(2, mode_val)
        print(f"    Fixed MaritalStatus: replaced numeric 2 with mode ({mode_val})")

# Fix Insurance - 'ensurance' -> 'insurance'
if 'Insurance' in df.columns:
    df['Insurance'] = df['Insurance'].replace('ensurance', 'insurance')
    print("    Fixed Insurance: 'ensurance' -> 'insurance'")

# Standardize yes/no columns
yes_no_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                df[col].nunique() <= 3 and 
                any(val in ['yes', 'no', 'Yes', 'No', 1, 0, True, False] 
                    for val in df[col].dropna().head(10).values)]

for col in yes_no_cols[:10]:  # Limit to first 10 to avoid too much output
    if col in df.columns:
        # Convert to lowercase strings, then standardize
        df[col] = df[col].astype(str).str.lower()
        df[col] = df[col].replace(['true', '1', 'yes'], 'yes')
        df[col] = df[col].replace(['false', '0', 'no'], 'no')
        df[col] = df[col].replace('nan', np.nan)

print(f"    Standardized {len(yes_no_cols)} yes/no columns")

# ============================================================================
# PHASE 5: Outlier Detection and Treatment
# ============================================================================
print("\n[PHASE 5] Detecting and treating outliers...")

outlier_flags = {}

def detect_outliers_iqr(series, multiplier=IQR_MULTIPLIER):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (series < lower_bound) | (series > upper_bound)

# Outlier detection for key numerical columns
outlier_columns = ['Weight', 'Height', 'BMI', 'Duration of hospitalization (days)']

for col in outlier_columns:
    if col in df.columns and df[col].dtype in [np.int64, np.float64]:
        outliers = detect_outliers_iqr(df[col])
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            # Create outlier flag
            flag_col = f'{col}_outlier'
            df[flag_col] = outliers.astype(int)
            outlier_flags[col] = outlier_count
            
            # Check if outliers are within medical ranges
            if col == 'Weight':
                valid_range = (WEIGHT_MIN_KG, WEIGHT_MAX_KG)
            elif col == 'Height':
                valid_range = (HEIGHT_MIN_CM, HEIGHT_MAX_CM)
            elif col == 'BMI':
                valid_range = (BMI_MIN, BMI_MAX)
            else:
                valid_range = None
            
            if valid_range:
                # Flag outliers outside medical range
                extreme_outliers = (df[col] < valid_range[0]) | (df[col] > valid_range[1])
                if extreme_outliers.sum() > 0:
                    print(f"    {col}: {extreme_outliers.sum()} extreme outliers outside medical range")
                    # Cap extreme values
                    df.loc[df[col] < valid_range[0], col] = valid_range[0]
                    df.loc[df[col] > valid_range[1], col] = valid_range[1]
                    print(f"      Capped extreme values to range {valid_range}")
            
            print(f"    {col}: {outlier_count} potential outliers detected (flagged)")

print(f"  Created {len(outlier_flags)} outlier flag columns")

# ============================================================================
# PHASE 6: Feature Engineering
# ============================================================================
print("\n[PHASE 6] Feature engineering...")

# Comorbidity count
print("  Creating comorbidity count...")
comorbidity_cols = [col for col in MEDICAL_HISTORY_COLUMNS if col in df.columns]
if comorbidity_cols:
    # Convert yes/no to binary
    df_comorb = df[comorbidity_cols].copy()
    for col in comorbidity_cols:
        df_comorb[col] = (df_comorb[col].astype(str).str.lower() == 'yes').astype(int)
    df['comorbidity_count'] = df_comorb.sum(axis=1)
    print(f"    Created comorbidity_count (mean: {df['comorbidity_count'].mean():.2f})")

# Medication count
print("  Creating medication count...")
medication_cols = [col for col in df.columns if 'Medication' in col or 'Drug' in col]
if medication_cols:
    df_med = df[medication_cols].copy()
    for col in medication_cols:
        if df_med[col].dtype == 'object':
            df_med[col] = (df_med[col].astype(str).str.lower() == 'yes').astype(int)
    df['medication_count'] = df_med.sum(axis=1)
    print(f"    Created medication_count (mean: {df['medication_count'].mean():.2f})")

# Complication count
print("  Creating complication count...")
complication_cols = [col for col in df.columns if 'complication' in col.lower() and 
                     col not in ['Complication post Discharge', 'Type of complication post discharge']]
if complication_cols:
    df_comp = df[complication_cols].copy()
    for col in complication_cols:
        if df_comp[col].dtype == 'object':
            df_comp[col] = (df_comp[col].astype(str).str.lower() == 'yes').astype(int)
    df['complication_count'] = df_comp.sum(axis=1)
    print(f"    Created complication_count (mean: {df['complication_count'].mean():.2f})")

# Lab value changes (deltas) - only for patients with both pre and post values
print("  Creating lab value deltas...")
if 'Pre-Creatinine' in df.columns and 'Creatinine_D1' in df.columns:
    mask = df['Pre-Creatinine'].notna() & df['Creatinine_D1'].notna()
    df.loc[mask, 'creatinine_delta_d1'] = df.loc[mask, 'Creatinine_D1'] - df.loc[mask, 'Pre-Creatinine']
    print(f"    Created creatinine_delta_d1 ({mask.sum()} patients)")

if 'Pre Na' in df.columns and 'Na day 1 post surgery' in df.columns:
    mask = df['Pre Na'].notna() & df['Na day 1 post surgery'].notna()
    df.loc[mask, 'na_delta_d1'] = df.loc[mask, 'Na day 1 post surgery'] - df.loc[mask, 'Pre Na']
    print(f"    Created na_delta_d1 ({mask.sum()} patients)")

if 'Pre HB' in df.columns and 'HB day 1 post surgery' in df.columns:
    mask = df['Pre HB'].notna() & df['HB day 1 post surgery'].notna()
    df.loc[mask, 'hb_delta_d1'] = df.loc[mask, 'HB day 1 post surgery'] - df.loc[mask, 'Pre HB']
    print(f"    Created hb_delta_d1 ({mask.sum()} patients)")

# Time-based features
print("  Creating time-based features...")
if 'Admission Date' in df.columns and 'Date of Surgery' in df.columns:
    df['days_admission_to_surgery'] = (df['Date of Surgery'] - df['Admission Date']).dt.days
    print(f"    Created days_admission_to_surgery (mean: {df['days_admission_to_surgery'].mean():.2f} days)")

# ============================================================================
# PHASE 7: Data Validation
# ============================================================================
print("\n[PHASE 7] Data validation...")

validation_issues = []

# Age validation
if 'Age' in df.columns:
    invalid_age = (df['Age'] < AGE_MIN) | (df['Age'] > AGE_MAX)
    if invalid_age.sum() > 0:
        validation_issues.append(f"Age: {invalid_age.sum()} values outside range [{AGE_MIN}, {AGE_MAX}]")
        print(f"  Age validation: {invalid_age.sum()} invalid values")

# Weight validation
if 'Weight' in df.columns:
    invalid_weight = (df['Weight'] < WEIGHT_MIN_KG) | (df['Weight'] > WEIGHT_MAX_KG)
    if invalid_weight.sum() > 0:
        validation_issues.append(f"Weight: {invalid_weight.sum()} values outside medical range")
        print(f"  Weight validation: {invalid_weight.sum()} invalid values")

# Height validation
if 'Height' in df.columns:
    invalid_height = (df['Height'] < HEIGHT_MIN_CM) | (df['Height'] > HEIGHT_MAX_CM)
    if invalid_height.sum() > 0:
        validation_issues.append(f"Height: {invalid_height.sum()} values outside medical range")
        print(f"  Height validation: {invalid_height.sum()} invalid values")

# BMI consistency check
if 'BMI' in df.columns and 'BMI_Category' in df.columns:
    # Recalculate BMI category and check consistency
    df['BMI_Category_calculated'] = df['BMI'].apply(categorize_bmi)
    inconsistent = df['BMI_Category'] != df['BMI_Category_calculated']
    inconsistent = inconsistent & df['BMI_Category'].notna() & df['BMI_Category_calculated'].notna()
    if inconsistent.sum() > 0:
        validation_issues.append(f"BMI_Category: {inconsistent.sum()} inconsistent with calculated BMI")
        print(f"  BMI consistency: {inconsistent.sum()} inconsistencies found")
        # Fix inconsistencies
        df.loc[inconsistent, 'BMI_Category'] = df.loc[inconsistent, 'BMI_Category_calculated']
    df = df.drop(columns=['BMI_Category_calculated'])

# Date logic validation
if 'Admission Date' in df.columns and 'Date of Surgery' in df.columns:
    invalid_dates = df['Date of Surgery'] < df['Admission Date']
    if invalid_dates.sum() > 0:
        validation_issues.append(f"Dates: {invalid_dates.sum()} surgeries before admission")
        print(f"  Date logic: {invalid_dates.sum()} invalid date sequences")

if validation_issues:
    print(f"  Total validation issues found: {len(validation_issues)}")
    with open(REPORTS_DIR / 'validation_issues.txt', 'w') as f:
        f.write("VALIDATION ISSUES\n")
        f.write("=" * 80 + "\n\n")
        for issue in validation_issues:
            f.write(issue + "\n")
    print("  Validation issues logged to: reports/validation_issues.txt")
else:
    print("  No validation issues found")

# ============================================================================
# PHASE 8: Prepare for Analysis
# ============================================================================
print("\n[PHASE 8] Preparing analysis-ready datasets...")

# Create separate datasets
df_numerical = df.select_dtypes(include=[np.number]).copy()
df_categorical = df.select_dtypes(include=['object', 'category']).copy()
df_mixed = df.copy()

print(f"  Numerical dataset: {df_numerical.shape}")
print(f"  Categorical dataset: {df_categorical.shape}")
print(f"  Mixed dataset: {df_mixed.shape}")

# ============================================================================
# PHASE 9: Export Cleaned Data
# ============================================================================
print("\n[PHASE 9] Exporting cleaned datasets...")

# Main cleaned dataset
df.to_excel(PROCESSED_DIR / 'data_cleaned.xlsx', index=False)
print("  Saved: data/processed/data_cleaned.xlsx")

# Subset datasets for lab analyses
df_pre_labs.to_excel(PROCESSED_DIR / 'data_pre_labs_subset.xlsx', index=False)
print(f"  Saved: data/processed/data_pre_labs_subset.xlsx ({len(df_pre_labs)} patients)")

df_post_labs.to_excel(PROCESSED_DIR / 'data_post_labs_subset.xlsx', index=False)
print(f"  Saved: data/processed/data_post_labs_subset.xlsx ({len(df_post_labs)} patients)")

df_discharge_labs.to_excel(PROCESSED_DIR / 'data_discharge_labs_subset.xlsx', index=False)
print(f"  Saved: data/processed/data_discharge_labs_subset.xlsx ({len(df_discharge_labs)} patients)")

# Follow-up subset
df_followup = df[df['follow_up_available'] == 1].copy()
df_followup.to_excel(PROCESSED_DIR / 'data_followup_subset.xlsx', index=False)
print(f"  Saved: data/processed/data_followup_subset.xlsx ({len(df_followup)} patients)")

# Analysis-ready datasets
df_numerical.to_excel(PROCESSED_DIR / 'data_numerical.xlsx', index=False)
print("  Saved: data/processed/data_numerical.xlsx")

df_categorical.to_excel(PROCESSED_DIR / 'data_categorical.xlsx', index=False)
print("  Saved: data/processed/data_categorical.xlsx")

# Generate final quality report
print("\nGenerating final data quality report...")
generate_quality_report(df, str(REPORTS_DIR / 'data_quality_report_final.txt'))
print("  Saved: reports/data_quality_report_final.txt")

print("\n" + "=" * 80)
print("DATA CLEANING PIPELINE COMPLETE!")
print("=" * 80)
print(f"\nFinal dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns removed: {df_backup.shape[1] - df.shape[1]}")
print(f"\nAll cleaned datasets and reports have been saved.")
