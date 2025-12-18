"""
Model Evaluation and Debugging Script
Tests the risk prediction model with different inputs to identify why predictions are constant
STANDALONE VERSION - No Streamlit dependencies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MODEL EVALUATION AND DEBUGGING")
print("=" * 80)

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
CONFIG_DIR = PROJECT_ROOT / 'config'

# Load data template
print("\n[1] Loading data template...")
data_path = DATA_DIR / 'data_cleaned.xlsx'
if not data_path.exists():
    print(f"    ERROR: Data file not found at {data_path}")
    sys.exit(1)

df_template = pd.read_excel(data_path)
print(f"    Template shape: {df_template.shape}")

# Load model
print("\n[2] Loading risk model...")

# List all available model files
all_pkl_files = list(MODELS_DIR.glob('*.pkl'))
print(f"    Available .pkl files in models/: {len(all_pkl_files)}")
for f in all_pkl_files:
    print(f"      - {f.name}")

# Try to find the risk model files
model_path = None
scaler_path = None
encoders_path = None
feature_columns_path = None
metadata_path = None

# Priority 1: Risk model files
if (MODELS_DIR / 'best_risk_model.pkl').exists():
    model_path = MODELS_DIR / 'best_risk_model.pkl'
    scaler_path = MODELS_DIR / 'risk_scaler.pkl'
    encoders_path = MODELS_DIR / 'risk_encoders.pkl'
    feature_columns_path = MODELS_DIR / 'risk_feature_columns.pkl'
    metadata_path = MODELS_DIR / 'risk_model_metadata.pkl'
    print(f"    Found risk model files")
# Priority 2: Logistic Regression
elif (MODELS_DIR / 'best_model_logistic_regression.pkl').exists():
    model_path = MODELS_DIR / 'best_model_logistic_regression.pkl'
    scaler_path = MODELS_DIR / 'scaler.pkl'
    encoders_path = MODELS_DIR / 'label_encoders.pkl'
    feature_columns_path = MODELS_DIR / 'feature_columns.pkl'
    print(f"    Found logistic regression model files")
# Priority 3: Any best_model_*.pkl
else:
    alt_models = list(MODELS_DIR.glob('best_model_*.pkl'))
    if alt_models:
        model_path = alt_models[0]
        scaler_path = MODELS_DIR / 'scaler.pkl'
        encoders_path = MODELS_DIR / 'label_encoders.pkl'
        feature_columns_path = MODELS_DIR / 'feature_columns.pkl'
        print(f"    Using: {model_path.name}")

if not model_path or not model_path.exists():
    print(f"    ERROR: No model file found!")
    print(f"    Available files: {[f.name for f in all_pkl_files]}")
    print("    Run: python scripts/risk_prediction_modeling.py")
    sys.exit(1)

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    encoders = joblib.load(encoders_path) if encoders_path.exists() else None
    feature_columns = joblib.load(feature_columns_path) if feature_columns_path.exists() else None
    metadata = joblib.load(metadata_path) if metadata_path.exists() else None
    
    print(f"    Model loaded: {model_path.name}")
    print(f"    Model type: {type(model).__name__}")
    if metadata:
        print(f"    ROC-AUC: {metadata.get('roc_auc', 'N/A'):.3f}")
    print(f"    Features expected: {len(feature_columns) if feature_columns else 'Unknown'}")
    if feature_columns:
        print(f"    First 10 features: {feature_columns[:10]}")
except Exception as e:
    error_msg = str(e)
    print(f"    ERROR loading model: {error_msg}")
    
    # Try loading logistic regression model instead
    if 'xgboost' in error_msg.lower():
        print(f"    XGBoost not available. Trying Logistic Regression model...")
        lr_model_path = MODELS_DIR / 'best_model_logistic_regression.pkl'
        if lr_model_path.exists():
            try:
                model = joblib.load(lr_model_path)
                scaler = joblib.load(MODELS_DIR / 'scaler.pkl') if (MODELS_DIR / 'scaler.pkl').exists() else None
                encoders = joblib.load(MODELS_DIR / 'label_encoders.pkl') if (MODELS_DIR / 'label_encoders.pkl').exists() else None
                feature_columns = joblib.load(MODELS_DIR / 'feature_columns.pkl') if (MODELS_DIR / 'feature_columns.pkl').exists() else None
                print(f"    SUCCESS: Loaded Logistic Regression model instead")
                print(f"    Model type: {type(model).__name__}")
                print(f"    Features expected: {len(feature_columns) if feature_columns else 'Unknown'}")
            except Exception as e2:
                print(f"    ERROR loading LR model: {str(e2)}")
                sys.exit(1)
        else:
            print(f"    Logistic Regression model not found either.")
            print(f"    Install XGBoost: pip install xgboost")
            sys.exit(1)
    else:
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Load VIF-selected features
print("\n[2.5] Loading VIF-selected features...")
try:
    sys.path.append(str(CONFIG_DIR))
    from selected_features_vif import SELECTED_FEATURES_VIF
    selected_features = SELECTED_FEATURES_VIF.copy()
    if 'Complication Post Surgery' in selected_features:
        selected_features.remove('Complication Post Surgery')
    print(f"    VIF-selected features: {len(selected_features)}")
except ImportError:
    print("    WARNING: Could not load VIF-selected features")
    selected_features = feature_columns if feature_columns else []

# Standalone feature preparation function (no Streamlit)
def prepare_features_standalone(input_data, df_template, scaler=None, encoders=None, feature_columns=None, selected_features=None):
    """Prepare features without Streamlit dependencies"""
    
    if feature_columns is None:
        raise ValueError("feature_columns must be provided")
    
    # Track user-provided values
    user_provided = set(input_data.keys())
    
    # Create DataFrame from input
    features_df = pd.DataFrame([input_data])
    
    # Calculate engineered features
    if 'comorbidity_count' not in features_df.columns or pd.isna(features_df['comorbidity_count'].iloc[0]):
        comorbidity_cols = ['Hypertension', 'DiabetesMellitus', 'Dyslipidemia', 'CAD History', 
                          'HF', 'COPD', 'CKD', 'Dialysis', 'Open heart surgery', 'AFib-tachycardia',
                          'PAD', 'Neurological/ Psychological disease', 'Gastrointestinal Disease',
                          'Endocrine Disease', 'Cancer', 'Allergy']
        comorbidity_count = 0
        for col in comorbidity_cols:
            if col in features_df.columns:
                val = features_df[col].iloc[0]
                if isinstance(val, str) and val.lower() in ['yes', 'true', '1']:
                    comorbidity_count += 1
                elif val == 1 or val is True:
                    comorbidity_count += 1
        features_df['comorbidity_count'] = comorbidity_count
    
    if 'medication_count' not in features_df.columns or pd.isna(features_df['medication_count'].iloc[0]):
        medication_cols = ['Current Medication', 'Antihypertensive', 'Antiplatelets', 'Anticoagulant',
                          'Antidiabetic', 'Thyroidal Medication', 'Antipsychotic', 'Betablocker',
                          'Cholesterol Lowering Drug', 'Diuratic', 'OtherMedication']
        medication_count = 0
        for col in medication_cols:
            if col in features_df.columns:
                val = features_df[col].iloc[0]
                if isinstance(val, str) and val.lower() in ['yes', 'true', '1']:
                    medication_count += 1
                elif val == 1 or val is True:
                    medication_count += 1
        features_df['medication_count'] = medication_count
    
    if ('BMI' not in features_df.columns or pd.isna(features_df['BMI'].iloc[0])) and 'Weight' in features_df.columns and 'Height' in features_df.columns:
        weight = features_df['Weight'].iloc[0]
        height = features_df['Height'].iloc[0]
        if not pd.isna(weight) and not pd.isna(height) and height > 0:
            features_df['BMI'] = weight / ((height/100) ** 2)
        else:
            features_df['BMI'] = df_template['BMI'].median() if 'BMI' in df_template.columns else 25.0
    
    if 'BMI_Category' not in features_df.columns or pd.isna(features_df['BMI_Category'].iloc[0]):
        if 'BMI' in features_df.columns:
            bmi = features_df['BMI'].iloc[0]
            if pd.isna(bmi) or bmi == 0:
                features_df['BMI_Category'] = 'normal'
            elif bmi < 18.5:
                features_df['BMI_Category'] = 'underweight'
            elif bmi < 25:
                features_df['BMI_Category'] = 'normal'
            elif bmi < 30:
                features_df['BMI_Category'] = 'overweight'
            else:
                features_df['BMI_Category'] = 'obese'
        else:
            features_df['BMI_Category'] = 'normal'
    
    # CRITICAL: Use feature_columns (what model expects), not selected_features
    # The model was trained on feature_columns, so we must use those
    if feature_columns:
        features_to_use = feature_columns.copy()
    elif selected_features:
        features_to_use = selected_features.copy()
    else:
        raise ValueError("No feature list available")
    
    if 'Complication Post Surgery' in features_to_use:
        features_to_use.remove('Complication Post Surgery')
    
    # Create X with only the features the model expects - PRESERVE USER VALUES
    X = pd.DataFrame(index=[0])
    for col in features_to_use:
        # CRITICAL: Check if user provided this value
        if col in features_df.columns:
            val = features_df[col].iloc[0]
            # If user provided and value is not NaN/empty, USE IT
            if col in user_provided:
                if pd.isna(val) or (isinstance(val, str) and val.lower() in ['', 'nan', 'none', 'unknown']):
                    # User provided but empty - fill with default
                    X[col] = np.nan
                else:
                    # USER PROVIDED VALUE - USE IT!
                    X[col] = val
            else:
                # Not provided by user - use what's in features_df (might be calculated)
                X[col] = val
        else:
            # Not in features_df at all - will fill with default
            X[col] = np.nan
    
    # DEBUG: Check if Age is preserved
    if 'Age' in X.columns:
        age_val = X['Age'].iloc[0]
        if 'Age' in user_provided:
            input_age = input_data.get('Age', 'N/A')
            if pd.isna(age_val) or age_val == 0:
                print(f"    DEBUG: Age lost! Input={input_age}, X[Age]={age_val}")
    
    # Fill missing values - ONLY for features NOT provided by user
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if X[col].isnull().any():
            # Only fill if user didn't provide this
            if col not in user_provided:
                if col in df_template.columns and df_template[col].dtype in [np.number]:
                    median_val = df_template[col].median()
                    X[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
                else:
                    X[col].fillna(0, inplace=True)
            # If user provided but it's NaN, still fill (edge case)
            elif pd.isna(X[col].iloc[0]):
                if col in df_template.columns and df_template[col].dtype in [np.number]:
                    median_val = df_template[col].median()
                    X[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
                else:
                    X[col].fillna(0, inplace=True)
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if X[col].isnull().any():
            # Only fill if user didn't provide this
            if col not in user_provided:
                if col in df_template.columns:
                    mode_val = df_template[col].mode()[0] if not df_template[col].mode().empty else 'Unknown'
                    X[col].fillna(mode_val, inplace=True)
                else:
                    X[col].fillna('Unknown', inplace=True)
            # If user provided but it's empty, fill
            elif pd.isna(X[col].iloc[0]) or str(X[col].iloc[0]).lower() in ['', 'nan', 'none', 'unknown']:
                if col in df_template.columns:
                    mode_val = df_template[col].mode()[0] if not df_template[col].mode().empty else 'Unknown'
                    X[col].fillna(mode_val, inplace=True)
                else:
                    X[col].fillna('Unknown', inplace=True)
    
    # Encode categorical variables
    X_encoded = X.copy()
    
    # DEBUG: Check Age after creating X_encoded
    if 'Age' in X_encoded.columns and 'Age' in user_provided:
        age_after_X = X_encoded['Age'].iloc[0]
        input_age = input_data.get('Age', 'N/A')
        if pd.isna(age_after_X) or age_after_X == 0:
            print(f"    DEBUG: Age is 0/NaN in X! Input={input_age}, X[Age]={age_after_X}")
        else:
            print(f"    DEBUG: Age OK in X! Input={input_age}, X[Age]={age_after_X}")
    
    categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col in X_encoded.columns:
            if encoders and col in encoders:
                try:
                    input_val = str(X_encoded[col].iloc[0])
                    if input_val in encoders[col].classes_:
                        X_encoded[col] = encoders[col].transform([input_val])[0]
                    else:
                        X_encoded[col] = 0
                except:
                    X_encoded[col] = 0
            else:
                le = LabelEncoder()
                if col in df_template.columns:
                    template_values = df_template[col].astype(str).unique()
                    le.fit(template_values)
                    input_val = str(X_encoded[col].iloc[0])
                    if input_val in le.classes_:
                        X_encoded[col] = le.transform([input_val])[0]
                    else:
                        X_encoded[col] = 0
                else:
                    X_encoded[col] = 0
    
    # Handle datetime
    datetime_cols = X_encoded.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        for col in datetime_cols:
            try:
                min_date = df_template[col].min()
                X_encoded[col] = (pd.to_datetime(X_encoded[col]) - min_date).dt.days
            except:
                X_encoded = X_encoded.drop(columns=[col])
    
    # Remove zero-variance - BUT DON'T REMOVE USER-PROVIDED FEATURES!
    numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
    zero_var_cols = []
    for col in numeric_cols:
        if col in X_encoded.columns:
            # Don't remove if user provided this value
            if col not in user_provided:
                if X_encoded[col].var() == 0 or X_encoded[col].nunique() <= 1:
                    zero_var_cols.append(col)
    
    if zero_var_cols:
        print(f"    DEBUG: Removing zero-variance columns: {zero_var_cols[:5]}")
        X_encoded = X_encoded.drop(columns=zero_var_cols)
    
    # DEBUG: Check Age after removing zero-variance
    if 'Age' in X_encoded.columns and 'Age' in user_provided:
        age_after_zero = X_encoded['Age'].iloc[0]
        input_age = input_data.get('Age', 'N/A')
        if pd.isna(age_after_zero) or age_after_zero == 0:
            print(f"    DEBUG: Age lost after zero-variance removal! Input={input_age}, X_encoded[Age]={age_after_zero}")
        elif 'Age' not in X_encoded.columns:
            print(f"    DEBUG: Age column removed! Input={input_age}")
    
    # Ensure all numeric
    non_numeric_cols = X_encoded.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        for col in non_numeric_cols:
            if col in X_encoded.columns:
                le = LabelEncoder()
                if col in df_template.columns:
                    template_values = df_template[col].astype(str).unique()
                    le.fit(template_values)
                    input_val = str(X_encoded[col].iloc[0])
                    if input_val in le.classes_:
                        X_encoded[col] = le.transform([input_val])[0]
                    else:
                        X_encoded[col] = 0
                else:
                    X_encoded[col] = 0
    
    # Reorder to match feature_columns - PRESERVE EXISTING VALUES
    if feature_columns:
        # DEBUG: Check Age and X_encoded state before reordering
        print(f"    DEBUG: Before reordering - X_encoded has {len(X_encoded.columns)} columns")
        if 'Age' in user_provided:
            input_age = input_data.get('Age', 'N/A')
            if 'Age' in X_encoded.columns:
                age_before = X_encoded['Age'].iloc[0]
                print(f"    DEBUG: Age in X_encoded.columns: Input={input_age}, X_encoded[Age]={age_before}")
            else:
                print(f"    DEBUG: Age NOT in X_encoded.columns! Input={input_age}")
                print(f"    DEBUG: X_encoded.columns: {list(X_encoded.columns)[:10]}")
        
        # First, add missing columns with defaults
        for col in feature_columns:
            if col not in X_encoded.columns:
                # DEBUG for Age
                if col == 'Age' and col in user_provided:
                    input_age = input_data.get('Age', 'N/A')
                    print(f"    DEBUG: Adding Age with default! Input={input_age}, this is WRONG!")
                if col in df_template.columns:
                    if df_template[col].dtype in [np.number]:
                        default_val = df_template[col].median() if not pd.isna(df_template[col].median()) else 0
                    else:
                        mode_val = df_template[col].mode()[0] if not df_template[col].mode().empty else 'Unknown'
                        if encoders and col in encoders:
                            if mode_val in encoders[col].classes_:
                                default_val = encoders[col].transform([mode_val])[0]
                            else:
                                default_val = 0
                        else:
                            default_val = 0
                    X_encoded[col] = default_val
                else:
                    X_encoded[col] = 0
        
        # CRITICAL: Reorder but preserve existing values
        # Create new DataFrame with correct order, preserving existing values
        X_reordered = pd.DataFrame(index=[0])
        for col in feature_columns:
            if col in X_encoded.columns:
                # PRESERVE the existing value - get the actual value
                # Use .values[0] instead of .iloc[0] to be safe
                val = X_encoded[col].values[0] if hasattr(X_encoded[col], 'values') else X_encoded[col].iloc[0]
                X_reordered[col] = val
                # DEBUG for Age
                if col == 'Age' and col in user_provided:
                    input_age = input_data.get('Age', 'N/A')
                    age_in_X_encoded = X_encoded[col].values[0] if hasattr(X_encoded[col], 'values') else X_encoded[col].iloc[0]
                    print(f"    DEBUG: Reordering Age - Input={input_age}, X_encoded[Age]={age_in_X_encoded}, val={val}")
                    if pd.isna(val) or val == 0:
                        print(f"    DEBUG: Age is 0/NaN! This is the problem!")
            else:
                # Shouldn't happen, but fill with 0
                X_reordered[col] = 0
                if col == 'Age' and col in user_provided:
                    input_age = input_data.get('Age', 'N/A')
                    print(f"    DEBUG: Age not in X_encoded.columns during reorder! Input={input_age}")
        X_encoded = X_reordered
    
    # Convert to float - PRESERVE VALUES
    # DEBUG: Check Age before float conversion
    if 'Age' in X_encoded.columns and 'Age' in user_provided:
        age_before_float = X_encoded['Age'].iloc[0]
        input_age = input_data.get('Age', 'N/A')
        print(f"    DEBUG: Age before float conversion: Input={input_age}, X_encoded[Age]={age_before_float}")
    
    try:
        X_encoded = X_encoded.astype(float)
    except:
        for col in X_encoded.columns:
            try:
                # For user-provided numeric columns, preserve the value
                if col in user_provided and X_encoded[col].dtype in [np.number]:
                    # Already numeric, keep it
                    pass
                else:
                    X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
            except:
                # Only set to 0 if not user-provided
                if col not in user_provided:
                    X_encoded[col] = 0
        X_encoded = X_encoded.astype(float)
    
    # DEBUG: Check Age after float conversion
    if 'Age' in X_encoded.columns and 'Age' in user_provided:
        age_after_float = X_encoded['Age'].iloc[0]
        input_age = input_data.get('Age', 'N/A')
        if pd.isna(age_after_float) or age_after_float == 0:
            print(f"    DEBUG: Age lost during float conversion! Input={input_age}, X_encoded[Age]={age_after_float}")
    
    # DEBUG: Check Age after all processing, before scaling
    if 'Age' in X_encoded.columns and 'Age' in user_provided:
        age_before_scale = X_encoded['Age'].iloc[0]
        input_age = input_data.get('Age', 'N/A')
        print(f"    DEBUG: Age before scaling: Input={input_age}, X_encoded[Age]={age_before_scale}")
    
    # Scale if needed
    if scaler:
        try:
            features_array = scaler.transform(X_encoded)
            # DEBUG: Check Age after scaling (first feature should be Age if feature_columns[0]=='Age')
            if feature_columns and feature_columns[0] == 'Age' and 'Age' in user_provided:
                input_age = input_data.get('Age', 'N/A')
                age_after_scale = features_array[0, 0]
                print(f"    DEBUG: Age after scaling: Input={input_age}, scaled={age_after_scale}")
        except Exception as e:
            print(f"    DEBUG: Scaling error: {e}")
            features_array = X_encoded.values.astype(float)
    else:
        features_array = X_encoded.values.astype(float)
        # DEBUG: Check Age in features_array
        if feature_columns and feature_columns[0] == 'Age' and 'Age' in user_provided:
            input_age = input_data.get('Age', 'N/A')
            age_in_array = features_array[0, 0]
            print(f"    DEBUG: Age in features_array (no scaling): Input={input_age}, array={age_in_array}")
    
    return features_array, X_encoded, user_provided

# Test scenarios
print("\n[3] Testing model with different input scenarios...")

test_scenarios = [
    {
        'name': 'Low Risk Patient',
        'data': {
            'Age': 30,
            'Gender': 'male',
            'Height': 175,
            'Weight': 70,
            'BMI': 22.9,
            'BMI_Category': 'normal',
            'Pre-BUN': 10.0,
            'Pre Na': 140.0,
            'Pre HB': 14.0,
            'Pre Platelet': 250.0,
            'Pre-Creatinine': 0.9,
            'Hypertension': 'no',
            'DiabetesMellitus': 'no',
            'CKD': 'no',
            'comorbidity_count': 0,
            'medication_count': 0,
            'Duration Of Surgery': 60,
            'days_admission_to_surgery': 1,
        }
    },
    {
        'name': 'High Risk Patient',
        'data': {
            'Age': 75,
            'Gender': 'male',
            'Height': 170,
            'Weight': 90,
            'BMI': 31.1,
            'BMI_Category': 'obese',
            'Pre-BUN': 25.0,
            'Pre Na': 135.0,
            'Pre HB': 9.0,
            'Pre Platelet': 150.0,
            'Pre-Creatinine': 2.5,
            'Hypertension': 'yes',
            'DiabetesMellitus': 'yes',
            'CKD': 'yes',
            'comorbidity_count': 5,
            'medication_count': 3,
            'Duration Of Surgery': 300,
            'days_admission_to_surgery': 5,
        }
    },
    {
        'name': 'Medium Risk Patient',
        'data': {
            'Age': 55,
            'Gender': 'female',
            'Height': 160,
            'Weight': 65,
            'BMI': 25.4,
            'BMI_Category': 'overweight',
            'Pre-BUN': 18.0,
            'Pre Na': 138.0,
            'Pre HB': 11.5,
            'Pre Platelet': 200.0,
            'Pre-Creatinine': 1.3,
            'Hypertension': 'yes',
            'DiabetesMellitus': 'no',
            'CKD': 'no',
            'comorbidity_count': 2,
            'medication_count': 1,
            'Duration Of Surgery': 150,
            'days_admission_to_surgery': 2,
        }
    }
]

results = []

for scenario in test_scenarios:
    print(f"\n  Testing: {scenario['name']}")
    print(f"    Input features: {len(scenario['data'])}")
    
    try:
        features_array, X_encoded, user_provided = prepare_features_standalone(
            scenario['data'], 
            df_template, 
            scaler, 
            encoders, 
            feature_columns,
            selected_features
        )
        
        print(f"    Prepared features shape: {features_array.shape}")
        print(f"    Expected features: {len(feature_columns) if feature_columns else 'Unknown'}")
        
        if features_array.shape[0] > 0:
            feature_values = features_array[0]
            unique_values = len(np.unique(feature_values))
            print(f"    Unique feature values: {unique_values} / {len(feature_values)}")
            
            # Show key feature values
            if feature_columns:
                print(f"    Key features (first 10):")
                for i, col in enumerate(feature_columns[:10]):
                    if i < len(feature_values):
                        print(f"      {col}: {feature_values[i]:.4f}")
            
            # Make prediction
            probability = model.predict_proba(features_array)[0, 1]
            prediction = model.predict(features_array)[0]
            risk_percentage = probability * 100
            
            print(f"    Probability: {probability:.4f}")
            print(f"    Risk Percentage: {risk_percentage:.2f}%")
            print(f"    Binary Prediction: {prediction}")
            
            results.append({
                'scenario': scenario['name'],
                'probability': probability,
                'risk_percentage': risk_percentage,
                'prediction': prediction,
                'features_array': features_array,
                'feature_values': feature_values,
                'X_encoded': X_encoded,
                'user_provided': user_provided
            })
        else:
            print("    ERROR: Empty features array!")
            
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# Compare results
print("\n" + "=" * 80)
print("COMPARISON OF PREDICTIONS")
print("=" * 80)

if len(results) >= 2:
    print("\n[4] Comparing predictions across scenarios...")
    
    for result in results:
        print(f"\n  {result['scenario']}:")
        print(f"    Risk: {result['risk_percentage']:.2f}%")
        print(f"    Probability: {result['probability']:.4f}")
    
    # Check if all predictions are the same
    risk_values = [r['risk_percentage'] for r in results]
    if len(set(risk_values)) == 1:
        print("\n  WARNING: All predictions are identical!")
        print(f"     All scenarios predict: {risk_values[0]:.2f}%")
        print("\n  This indicates a problem with feature preparation.")
    else:
        print("\n  OK: Predictions vary across scenarios (expected behavior)")
    
    # Compare feature arrays
    print("\n[5] Comparing feature values across scenarios...")
    
    if feature_columns and len(results) >= 2:
        features_1 = results[0]['feature_values']
        features_2 = results[1]['feature_values']
        
        differences = []
        for i, (val1, val2) in enumerate(zip(features_1, features_2)):
            if i < len(feature_columns):
                col_name = feature_columns[i]
                diff = abs(val1 - val2)
                if diff > 0.001:
                    differences.append((col_name, val1, val2, diff))
        
        print(f"    Features with differences: {len(differences)} / {len(feature_columns)}")
        
        if len(differences) == 0:
            print("\n  CRITICAL: No feature differences detected!")
            print("     This means all inputs are producing identical feature arrays.")
            print("     The model is receiving the same features regardless of input.")
        else:
            print(f"\n  Top 20 features with largest differences:")
            differences_sorted = sorted(differences, key=lambda x: x[3], reverse=True)[:20]
            for col_name, val1, val2, diff in differences_sorted:
                print(f"    {col_name}: {val1:.4f} vs {val2:.4f} (diff: {diff:.4f})")
            
            # Check which user-provided features are actually different
            print(f"\n  Checking if user-provided features are preserved:")
            user_features_1 = results[0]['user_provided']
            user_features_2 = results[1]['user_provided']
            
            # Check Age specifically
            if 'Age' in feature_columns:
                age_idx = feature_columns.index('Age')
                age_1 = features_1[age_idx]
                age_2 = features_2[age_idx]
                print(f"    Age in features: {age_1:.2f} vs {age_2:.2f} (input: 30 vs 75)")
                if abs(age_1 - age_2) < 0.1:
                    print("    WARNING: Age is NOT being preserved correctly!")

# Debug feature preparation
print("\n" + "=" * 80)
print("DETAILED FEATURE PREPARATION DEBUG")
print("=" * 80)

if results:
    print("\n[6] Checking feature preparation for first scenario...")
    result = results[0]
    
    print(f"\n  User-provided features: {len(result['user_provided'])}")
    print(f"    {list(result['user_provided'])[:10]}")
    
    if feature_columns:
        print(f"\n  Model expects {len(feature_columns)} features")
        
        # Check which user features match
        matching = [col for col in result['user_provided'] if col in feature_columns]
        print(f"  User features that match model: {len(matching)}")
        print(f"    {matching[:10]}")
        
        # Check X_encoded DataFrame
        if 'X_encoded' in result:
            X_df = result['X_encoded']
            print(f"\n  X_encoded DataFrame shape: {X_df.shape}")
            print(f"  X_encoded columns (first 20): {list(X_df.columns)[:20]}")
            
            # Check if user values are in X_encoded
            print(f"\n  Checking if user values are preserved in X_encoded:")
            for col in ['Age', 'BMI', 'Pre-Creatinine', 'comorbidity_count']:
                if col in X_df.columns:
                    val = X_df[col].iloc[0]
                    input_val = test_scenarios[0]['data'].get(col, 'N/A')
                    print(f"    {col}: X_encoded={val:.4f}, input={input_val}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nIf all predictions are identical (2%), the likely causes are:")
print("1. User inputs are being overwritten with template defaults")
print("2. Feature preparation is not preserving user values")
print("3. Model is receiving constant feature arrays regardless of input")
print("4. Feature encoding is converting all inputs to the same values")
print("\nCheck the feature differences above to identify the issue.")
