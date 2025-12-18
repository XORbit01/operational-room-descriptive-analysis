"""
VIF-based Feature Selection for Risk Prediction Model
Removes redundant features using Variance Inflation Factor (VIF) analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
import sys

# Add config directory to path
sys.path.append(str(Path(__file__).parent.parent / 'config'))
try:
    from modeling_config import VIF_THRESHOLD, VIF_MAX_ITERATIONS
except ImportError:
    # Default values if config doesn't exist yet
    VIF_THRESHOLD = 10.0
    VIF_MAX_ITERATIONS = 10

print("=" * 80)
print("VIF-BASED FEATURE SELECTION")
print("=" * 80)

# Set up paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'
CONFIG_DIR = Path(__file__).parent.parent / 'config'

# ============================================================================
# 1. Load Data and Select Pre-Surgery Features
# ============================================================================

print("\n[1] Loading data and selecting pre-surgery features...")
df = pd.read_excel(PROCESSED_DIR / 'data_cleaned.xlsx')
print(f"    Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# Identify target variable
TARGET_VARIABLE = 'Complication Post Surgery'

# Define features to exclude (post-surgery, outcomes, identifiers)
exclude_categories = {
    'Post-Surgery Outcomes': [
        'Complication During Surgery', 'Cardiac Complication',
        'Pulmonary complication', 'Renal complication', 'Neurological complication',
        'Infection of the surgical site', 'Other Type of complication',
        'Death post surgery during hospitalization', 'Complication Post Surgery',
    ],
    'Post-Surgery Labs': [
        'BUN day 1 post surgery', 'Creatinine_D1', 'Na day 1 post surgery',
        'HB day 1 post surgery', 'Platelet day 1 post surgery',
        'creatinine_delta_d1', 'na_delta_d1', 'hb_delta_d1',
    ],
    'Discharge Information': [
        'Creatinine before Discharge', 'BUN before Discharge',
        'Na before Discharge', 'HB before discharge', 'Platelet before Discharge',
        'Duration of hospitalization (days)', 'Duration in intensive care unit (days)',
    ],
    'Post-Discharge Outcomes': [
        'Complication post Discharge', 'ER Visit', 'Readmission due to OR',
        'Infection or inflammation', 'Redo surgery', 'Death post discharge',
    ],
    'Non-Predictive Identifiers': [
        'Patient name', 'ID number', 'Phone number', 'Physician Name',
        'Descriptions', 'Pathology description',
    ],
    'Complication Types': [
        'Type of cardiac Complication', 'Type of pulmonary complication',
        'Type of renal complication', 'Type of nurologic complication',
        'Type of complication post discharge',
    ],
    'Lab Availability Flags': [
        'has_pre_labs', 'has_post_labs', 'has_discharge_labs', 'follow_up_available',
    ],
    'Outlier Flags': [
        col for col in df.columns if '_outlier' in col
    ],
}

# Collect all features to exclude
excluded_features = []
for category, features in exclude_categories.items():
    excluded_features.extend([f for f in features if f in df.columns])

# Get pre-surgery features (all columns except excluded)
pre_surgery_features = [col for col in df.columns if col not in excluded_features and col != TARGET_VARIABLE]

print(f"    Pre-surgery features: {len(pre_surgery_features)}")
print(f"    Excluded features: {len(excluded_features)}")

# Create working dataset with pre-surgery features
df_work = df[pre_surgery_features + [TARGET_VARIABLE]].copy()

# ============================================================================
# 2. Handle Missing Values and Encode Categorical Variables
# ============================================================================

print("\n[2] Handling missing values and encoding categorical variables...")

# Separate numerical and categorical columns
numerical_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_work.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target from numerical if present
if TARGET_VARIABLE in numerical_cols:
    numerical_cols.remove(TARGET_VARIABLE)

print(f"    Numerical features: {len(numerical_cols)}")
print(f"    Categorical features: {len(categorical_cols)}")

# Handle missing values in numerical columns
for col in numerical_cols:
    if df_work[col].isnull().sum() > 0:
        df_work[col].fillna(df_work[col].median(), inplace=True)

# Handle missing values in categorical columns
for col in categorical_cols:
    if df_work[col].isnull().sum() > 0:
        mode_val = df_work[col].mode()[0] if not df_work[col].mode().empty else 'Unknown'
        df_work[col].fillna(mode_val, inplace=True)

# One-hot encode categorical variables for VIF calculation
print("    One-hot encoding categorical variables...")
df_encoded = df_work[numerical_cols].copy()
label_encoders = {}

# Ensure numerical columns are numeric
for col in df_encoded.columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')

for col in categorical_cols:
    if col in df_work.columns:
        # Get unique values
        unique_vals = df_work[col].dropna().unique()
        
        # For binary or low cardinality (<10), use one-hot encoding
        if len(unique_vals) <= 10:
            dummies = pd.get_dummies(df_work[col], prefix=col, dummy_na=False)
            # Ensure dummies are numeric
            dummies = dummies.astype(float)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
        else:
            # For high cardinality, use label encoding
            le = LabelEncoder()
            encoded_vals = le.fit_transform(df_work[col].astype(str))
            df_encoded[col] = encoded_vals.astype(float)
            label_encoders[col] = le

# Fill any NaN values with median
df_encoded = df_encoded.fillna(df_encoded.median())

# Replace infinite values
df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
df_encoded = df_encoded.fillna(df_encoded.median())

# Ensure all columns are float
df_encoded = df_encoded.astype(float)

print(f"    Encoded dataset shape: {df_encoded.shape}")

# ============================================================================
# 3. Calculate VIF and Remove Redundant Features
# ============================================================================

print("\n[3] Calculating VIF and removing redundant features...")
print(f"    VIF threshold: {VIF_THRESHOLD}")
print(f"    Max iterations: {VIF_MAX_ITERATIONS}")

def calculate_vif(X):
    """Calculate VIF for all features"""
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Ensure all columns are numeric and handle NaN/inf values
    X_clean = X.copy()
    
    # Convert all columns to numeric, coercing errors to NaN
    # Use a copy of column names to avoid iteration issues
    cols_to_drop = []
    current_cols = list(X_clean.columns)
    
    for col in current_cols:
        # Check if column still exists (might have been dropped)
        if col not in X_clean.columns:
            continue
        try:
            # Access column safely
            series = X_clean[col]
            X_clean[col] = pd.to_numeric(series, errors='coerce')
        except KeyError:
            # Column doesn't exist, skip it
            continue
        except (TypeError, ValueError) as e:
            # If conversion fails, mark for dropping
            if col not in cols_to_drop:
                cols_to_drop.append(col)
        except Exception as e:
            # If still fails, mark for dropping
            if col not in cols_to_drop:
                cols_to_drop.append(col)
    
    # Drop columns that failed conversion
    if cols_to_drop:
        X_clean = X_clean.drop(columns=[col for col in cols_to_drop if col in X_clean.columns])
    
    # Remove columns with all NaN
    X_clean = X_clean.dropna(axis=1, how='all')
    
    # Check for constant columns
    constant_cols = []
    for col in list(X_clean.columns):
        try:
            if col not in X_clean.columns:
                continue
            if X_clean[col].nunique() <= 1:
                constant_cols.append(col)
            elif X_clean[col].var() == 0 or pd.isna(X_clean[col].var()):
                constant_cols.append(col)
        except (KeyError, AttributeError):
            continue
        except Exception:
            constant_cols.append(col)
    
    if constant_cols:
        X_clean = X_clean.drop(columns=[col for col in constant_cols if col in X_clean.columns])
    
    # Fill remaining NaN values with column median
    for col in list(X_clean.columns):
        if col not in X_clean.columns:
            continue
        try:
            median_val = X_clean[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X_clean[col] = X_clean[col].fillna(median_val)
        except (KeyError, AttributeError):
            continue
    
    # Replace infinite values
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    for col in list(X_clean.columns):
        if col not in X_clean.columns:
            continue
        try:
            median_val = X_clean[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X_clean[col] = X_clean[col].fillna(median_val)
        except (KeyError, AttributeError):
            continue
    
    # Ensure all values are finite and numeric
    try:
        X_clean = X_clean.astype(float)
    except Exception:
        # If conversion fails, try column by column
        for col in list(X_clean.columns):
            if col in X_clean.columns:
                try:
                    X_clean[col] = X_clean[col].astype(float)
                except:
                    X_clean = X_clean.drop(columns=[col])
    
    # Check if we have enough features
    if len(X_clean.columns) < 2:
        original_cols = X.columns if hasattr(X, 'columns') else list(range(X.shape[1]))
        return pd.DataFrame({"Feature": original_cols, "VIF": [np.nan] * len(original_cols)})
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_clean.columns.tolist()
    vif_values = []
    
    # Convert to numpy array for VIF calculation
    X_array = X_clean.values.astype(float)
    
    for i in range(len(X_clean.columns)):
        try:
            vif_val = variance_inflation_factor(X_array, i)
            # Handle infinite or NaN VIF values
            if np.isinf(vif_val) or np.isnan(vif_val):
                vif_val = np.nan
            vif_values.append(float(vif_val))
        except Exception as e:
            # If VIF calculation fails for a feature, set to NaN
            vif_values.append(np.nan)
    
    vif_data["VIF"] = vif_values
    
    # Add back any columns that were removed (with NaN VIF)
    original_cols = X.columns if hasattr(X, 'columns') else list(range(X.shape[1]))
    removed_cols = set(original_cols) - set(X_clean.columns)
    if removed_cols:
        removed_df = pd.DataFrame({
            "Feature": list(removed_cols),
            "VIF": [np.nan] * len(removed_cols)
        })
        vif_data = pd.concat([vif_data, removed_df], ignore_index=True)
    
    return vif_data

# Initial VIF calculation
vif_df = calculate_vif(df_encoded)
vif_df = vif_df.sort_values('VIF', ascending=False)

print(f"\n    Initial features: {len(df_encoded.columns)}")
print(f"    Features with VIF > {VIF_THRESHOLD}: {(vif_df['VIF'] > VIF_THRESHOLD).sum()}")

# Iteratively remove features with high VIF
removed_features = []
iteration = 0
df_vif = df_encoded.copy()

# Ensure df_vif is numeric before VIF calculation
for col in list(df_vif.columns):
    if col in df_vif.columns:
        try:
            series = df_vif[col]
            df_vif[col] = pd.to_numeric(series, errors='coerce')
        except (KeyError, TypeError, ValueError):
            # Skip columns that can't be converted
            continue

# Fill NaN values with median
for col in list(df_vif.columns):
    if col in df_vif.columns:
        try:
            # Get median as scalar value
            median_val = float(df_vif[col].median())
            if pd.isna(median_val) or np.isnan(median_val):
                median_val = 0.0
            df_vif[col] = df_vif[col].fillna(median_val)
        except (KeyError, AttributeError, TypeError, ValueError):
            # If median calculation fails, use 0.0
            try:
                df_vif[col] = df_vif[col].fillna(0.0)
            except:
                continue

while iteration < VIF_MAX_ITERATIONS:
    vif_df = calculate_vif(df_vif)
    # Sort by VIF, putting NaN values last
    vif_df = vif_df.sort_values('VIF', ascending=False, na_position='last')
    
    # Filter out NaN VIF values
    high_vif = vif_df[(vif_df['VIF'] > VIF_THRESHOLD) & (vif_df['VIF'].notna())]
    
    if len(high_vif) == 0:
        print(f"\n    Iteration {iteration + 1}: No features with VIF > {VIF_THRESHOLD}")
        break
    
    # Remove the feature with highest VIF
    feature_to_remove = high_vif.iloc[0]['Feature']
    removed_features.append({
        'feature': feature_to_remove,
        'vif': high_vif.iloc[0]['VIF'],
        'iteration': iteration + 1
    })
    
    if feature_to_remove in df_vif.columns:
        df_vif = df_vif.drop(columns=[feature_to_remove])
        print(f"    Iteration {iteration + 1}: Removed '{feature_to_remove}' (VIF: {high_vif.iloc[0]['VIF']:.2f})")
    else:
        print(f"    Iteration {iteration + 1}: Feature '{feature_to_remove}' not found, skipping")
        break
    
    iteration += 1

# Final VIF calculation
final_vif = calculate_vif(df_vif)
final_vif = final_vif.sort_values('VIF', ascending=False)

print(f"\n    Final features: {len(df_vif.columns)}")
print(f"    Features removed: {len(removed_features)}")
print(f"    Max VIF remaining: {final_vif['VIF'].max():.2f}")

# ============================================================================
# 4. Handle Special Cases (Clinical Relevance)
# ============================================================================

print("\n[4] Applying clinical relevance rules...")

# Map encoded features back to original features
selected_features = []
feature_mapping = {}

# Check which original features are represented in selected encoded features
for col in df_vif.columns:
    # Check if it's a one-hot encoded feature
    if '_' in col and any(col.startswith(cat_col + '_') for cat_col in categorical_cols):
        # Extract original categorical feature name
        for cat_col in categorical_cols:
            if col.startswith(cat_col + '_'):
                if cat_col not in selected_features:
                    selected_features.append(cat_col)
                break
    elif col in numerical_cols:
        selected_features.append(col)
    elif col in categorical_cols:
        selected_features.append(col)

# Special handling for highly correlated pairs
# Based on correlation analysis: Weight ↔ BMI (r=0.82)
# Keep BMI, remove Weight if both present
if 'Weight' in selected_features and 'BMI' in selected_features:
    if 'Weight' in df_vif.columns or 'BMI' in df_vif.columns:
        # Check which has lower VIF
        weight_vif = final_vif[final_vif['Feature'] == 'Weight']['VIF'].values
        bmi_vif = final_vif[final_vif['Feature'] == 'BMI']['VIF'].values
        
        if len(weight_vif) > 0 and len(bmi_vif) > 0:
            if weight_vif[0] > bmi_vif[0]:
                if 'Weight' in selected_features:
                    selected_features.remove('Weight')
                    print("    Removed 'Weight' (keeping BMI due to high correlation)")
            else:
                if 'BMI' in selected_features:
                    selected_features.remove('BMI')
                    print("    Removed 'BMI' (keeping Weight due to high correlation)")

# Remove data leakage features
# complication_count correlates 0.93 with target - this is data leakage
if 'complication_count' in selected_features:
    selected_features.remove('complication_count')
    print("    Removed 'complication_count' (data leakage - correlates 0.93 with target)")

# Ensure we have the target variable
if TARGET_VARIABLE not in selected_features:
    selected_features.append(TARGET_VARIABLE)

# Filter to only features that exist in original dataset
selected_features = [f for f in selected_features if f in df.columns]

print(f"    Final selected features: {len(selected_features)}")

# ============================================================================
# 5. Generate Report
# ============================================================================

print("\n[5] Generating feature selection report...")

report_lines = []
report_lines.append("=" * 80)
report_lines.append("VIF-BASED FEATURE SELECTION REPORT")
report_lines.append("=" * 80)
report_lines.append(f"\nDate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"\nVIF Threshold: {VIF_THRESHOLD}")
report_lines.append(f"Max Iterations: {VIF_MAX_ITERATIONS}")

report_lines.append(f"\n\nINITIAL STATE:")
report_lines.append(f"  Total pre-surgery features: {len(pre_surgery_features)}")
report_lines.append(f"  Features after encoding: {len(df_encoded.columns)}")

report_lines.append(f"\n\nFEATURES REMOVED:")
report_lines.append("-" * 80)
if removed_features:
    for rem in removed_features:
        report_lines.append(f"  Iteration {rem['iteration']}: {rem['feature']} (VIF: {rem['vif']:.2f})")
else:
    report_lines.append("  No features removed (all VIF values below threshold)")

report_lines.append(f"\n\nFINAL SELECTED FEATURES ({len(selected_features)}):")
report_lines.append("-" * 80)
for i, feat in enumerate(selected_features, 1):
    if feat != TARGET_VARIABLE:
        # Try to find VIF value if it was in encoded dataset
        vif_val = "N/A"
        if feat in final_vif['Feature'].values:
            vif_val = f"{final_vif[final_vif['Feature'] == feat]['VIF'].values[0]:.2f}"
        report_lines.append(f"  {i:3d}. {feat:50s} (VIF: {vif_val})")

report_lines.append(f"\n\nFEATURE REDUCTION:")
report_lines.append(f"  Original features: {len(pre_surgery_features)}")
report_lines.append(f"  Selected features: {len([f for f in selected_features if f != TARGET_VARIABLE])}")
reduction_pct = (1 - len([f for f in selected_features if f != TARGET_VARIABLE]) / len(pre_surgery_features)) * 100
report_lines.append(f"  Reduction: {reduction_pct:.1f}%")

# Save report
with open(REPORTS_DIR / 'feature_selection_vif_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"    Saved report: reports/feature_selection_vif_report.txt")

# ============================================================================
# 6. Save Selected Features
# ============================================================================

print("\n[6] Saving selected features...")

# Save as Python list for import
selected_features_list = [f for f in selected_features if f != TARGET_VARIABLE]

with open(CONFIG_DIR / 'selected_features_vif.py', 'w', encoding='utf-8') as f:
    f.write('"""\n')
    f.write('VIF-Selected Features for Risk Prediction Model\n')
    f.write('Features selected using Variance Inflation Factor (VIF) analysis\n')
    f.write(f'VIF Threshold: {VIF_THRESHOLD}\n')
    f.write(f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write('"""\n\n')
    f.write('SELECTED_FEATURES_VIF = [\n')
    for feat in selected_features_list:
        f.write(f"    '{feat}',\n")
    f.write(']\n')

print(f"    Saved: config/selected_features_vif.py")

# Also save as pickle for easy loading
import joblib
joblib.dump(selected_features_list, CONFIG_DIR / 'selected_features_vif.pkl')
print(f"    Saved: config/selected_features_vif.pkl")

print("\n" + "=" * 80)
print("VIF-BASED FEATURE SELECTION COMPLETE!")
print("=" * 80)
print(f"\nSelected {len(selected_features_list)} features")
print(f"Reduced from {len(pre_surgery_features)} features ({reduction_pct:.1f}% reduction)")
print(f"\nNext step: Run scripts/risk_prediction_modeling.py to train models with selected features")
