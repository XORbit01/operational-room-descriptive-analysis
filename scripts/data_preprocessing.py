"""
Data preprocessing for correlation analysis
Handles encoding of categorical variables and prepares data for correlation analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATA PREPROCESSING FOR CORRELATION ANALYSIS")
print("=" * 80)

# Set up paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'

# Load cleaned data
print("\nLoading cleaned dataset...")
df = pd.read_excel(PROCESSED_DIR / 'data_cleaned.xlsx')
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# Encode Categorical Variables
# ============================================================================
print("\n[ENCODING] Encoding categorical variables...")

df_encoded = df.copy()
encoding_log = []

# Binary encoding for yes/no variables
print("  Binary encoding yes/no variables...")
yes_no_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 3:
            # Check if it's a yes/no column
            str_vals = [str(v).lower() for v in unique_vals]
            if any(v in ['yes', 'no', 'true', 'false', '1', '0'] for v in str_vals):
                yes_no_cols.append(col)

for col in yes_no_cols:
    if col not in ['Type of complication post discharge', 'Type of cancer', 
                   'Type of Neurological/ psychological disease', 
                   'Type of Gastrointestinal Disease', 'Type of Endocrine Disease']:
        df_encoded[col] = (df_encoded[col].astype(str).str.lower() == 'yes').astype(int)
        encoding_log.append(f"Binary encoded: {col}")

print(f"    Binary encoded {len(yes_no_cols)} columns")

# One-hot encoding for nominal categories with few levels (<10)
print("  One-hot encoding nominal categories...")
nominal_cols = ['Gender', 'Governorate', 'MaritalStatus', 'Insurance', 
                'BMI_Category', 'SmokingStatus', 'BloodGroup',
                'Emergency Status of surgery', 'Anesthesia type', 'Way Of Anesthesia']

one_hot_encoded = []
for col in nominal_cols:
    if col in df_encoded.columns:
        unique_count = df_encoded[col].nunique()
        if 2 <= unique_count <= 10:  # Between 2 and 10 unique values
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=False)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            one_hot_encoded.append(col)
            encoding_log.append(f"One-hot encoded: {col} ({unique_count} categories)")

print(f"    One-hot encoded {len(one_hot_encoded)} columns")

# Label encoding for ordinal categories
print("  Label encoding ordinal categories...")
# BMI_Category is already one-hot encoded, but if we need ordinal:
# We could create an ordinal version if needed for certain analyses

# Handle high cardinality categorical variables
print("  Handling high cardinality categorical variables...")
high_card_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        unique_count = df[col].nunique()
        if unique_count > 10 and col not in ['Descriptions', 'Pathology description']:
            high_card_cols.append((col, unique_count))

if high_card_cols:
    print(f"    Found {len(high_card_cols)} high cardinality columns:")
    for col, count in high_card_cols[:5]:  # Show first 5
        print(f"      {col}: {count} unique values")
        # For correlation analysis, we might exclude these or use target encoding
        # For now, we'll keep them as-is and exclude from correlation matrix

# ============================================================================
# Prepare Correlation-Ready Dataset
# ============================================================================
print("\n[PREPARATION] Preparing correlation-ready dataset...")

# Select only numerical columns for correlation
correlation_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Remove identifier-like columns
correlation_cols = [col for col in correlation_cols if 
                   'ID' not in col and 'name' not in col.lower() and 
                   'phone' not in col.lower()]

# Remove columns with no variance
for col in correlation_cols:
    if df_encoded[col].nunique() <= 1:
        correlation_cols.remove(col)

print(f"    Selected {len(correlation_cols)} numerical columns for correlation")

# Create correlation dataset
df_correlation = df_encoded[correlation_cols].copy()

# Handle multicollinearity - remove highly correlated features
print("  Checking for multicollinearity...")
corr_matrix = df_correlation.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Find pairs with correlation > 0.95
high_corr_pairs = []
for col in upper_triangle.columns:
    high_corr = upper_triangle[col][upper_triangle[col] > 0.95]
    if len(high_corr) > 0:
        for idx, val in high_corr.items():
            high_corr_pairs.append((col, idx, val))

if high_corr_pairs:
    print(f"    Found {len(high_corr_pairs)} highly correlated pairs (>0.95):")
    # Remove one from each pair (prefer to keep the more informative one)
    cols_to_remove = set()
    for col1, col2, corr_val in high_corr_pairs[:10]:  # Show first 10
        print(f"      {col1} <-> {col2}: {corr_val:.3f}")
        # Remove the one with fewer unique values or more missing
        if df_correlation[col1].nunique() < df_correlation[col2].nunique():
            cols_to_remove.add(col1)
        else:
            cols_to_remove.add(col2)
    
    df_correlation = df_correlation.drop(columns=list(cols_to_remove))
    print(f"    Removed {len(cols_to_remove)} columns due to multicollinearity")

# Handle remaining missing values for correlation (pairwise deletion will be used)
print(f"  Final correlation dataset: {df_correlation.shape}")
print(f"    Missing values: {df_correlation.isnull().sum().sum()} total cells")

# Save encoded dataset
print("\n[Saving] Saving preprocessed datasets...")
df_encoded.to_excel(PROCESSED_DIR / 'data_encoded.xlsx', index=False)
print("  Saved: data/processed/data_encoded.xlsx")

df_correlation.to_excel(PROCESSED_DIR / 'data_correlation_ready.xlsx', index=False)
print("  Saved: data/processed/data_correlation_ready.xlsx")

# Save encoding log
with open(REPORTS_DIR / 'encoding_log.txt', 'w') as f:
    f.write("ENCODING LOG\n")
    f.write("=" * 80 + "\n\n")
    for log_entry in encoding_log:
        f.write(log_entry + "\n")

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE!")
print("=" * 80)