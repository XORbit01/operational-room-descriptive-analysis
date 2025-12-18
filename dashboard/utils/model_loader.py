"""
Model loading utilities for prediction in dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def get_model_path():
    """Get the path to the saved model"""
    possible_paths = [
        Path(__file__).parent.parent.parent / 'models',
        Path.cwd() / 'models',
        Path('models'),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return Path(__file__).parent.parent.parent / 'models'

MODELS_DIR = get_model_path()

@st.cache_resource
def load_model():
    """Load the best trained model"""
    # Try to find the best model file
    model_files = list(MODELS_DIR.glob('best_model_*.pkl'))
    
    if not model_files:
        return None, None, None, None, "No model files found"
    
    # Load the first model found (or you can specify which one)
    model_path = model_files[0]
    
    try:
        model = joblib.load(model_path)
        
        # Try to load scaler, encoders, and feature columns
        scaler_path = MODELS_DIR / 'scaler.pkl'
        encoders_path = MODELS_DIR / 'label_encoders.pkl'
        feature_columns_path = MODELS_DIR / 'feature_columns.pkl'
        
        scaler = None
        encoders = None
        feature_columns = None
        
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)
        
        if feature_columns_path.exists():
            feature_columns = joblib.load(feature_columns_path)
        
        model_name = model_path.stem.replace('best_model_', '').replace('_', ' ').title()
        return model, scaler, encoders, feature_columns, None
        
    except Exception as e:
        return None, None, None, None, f"Error loading model: {str(e)}"

def prepare_features_for_prediction(input_data, df_template, scaler=None, encoders=None, feature_columns=None):
    """
    Prepare input data for prediction by matching the training data format exactly.
    This replicates the exact preprocessing steps from risk_prediction_modeling.py
    
    Args:
        input_data: Dictionary of input features
        df_template: DataFrame with the same structure as training data
        scaler: Fitted scaler (if model requires scaling)
        encoders: Dictionary of label encoders
        feature_columns: List of feature columns the model expects (in order) - CRITICAL
    
    Returns:
        Prepared feature array ready for prediction
    """
    from sklearn.preprocessing import LabelEncoder
    
    # CRITICAL: We must use only the features the model was trained with
    if feature_columns is None:
        st.error("Feature columns not provided! Model cannot make predictions.")
        raise ValueError("feature_columns must be provided")
    
    # CRITICAL: Start with user input_data, don't overwrite with template defaults
    # Track which values were provided by user (not defaults) - BEFORE any modifications
    user_provided = set(input_data.keys())
    
    # Create a DataFrame from input data - preserve user values
    features_df = pd.DataFrame([input_data])
    
    # Calculate engineered features if not provided (or if NaN)
    if 'comorbidity_count' not in features_df.columns or pd.isna(features_df['comorbidity_count'].iloc[0]):
        comorbidity_cols = ['Hypertension', 'DiabetesMellitus', 'Dyslipidemia', 'CAD History', 
                          'HF', 'COPD', 'CKD', 'Dialysis', 'Open heart surgery', 'AFib-tachycardia',
                          'PAD', 'Neurological/ Psychological disease', 'Gastrointestinal Disease',
                          'Endocrine Disease', 'Cancer', 'Allergy']
        comorbidity_count = 0
        for col in comorbidity_cols:
            if col in features_df.columns:
                val = features_df[col].iloc[0]
                if isinstance(val, str):
                    if val.lower() in ['yes', 'true', '1']:
                        comorbidity_count += 1
                elif val == 1 or val is True:
                    comorbidity_count += 1
        features_df['comorbidity_count'] = comorbidity_count
    
    # Calculate medication_count if not provided
    if 'medication_count' not in features_df.columns or pd.isna(features_df['medication_count'].iloc[0]):
        medication_cols = ['Current Medication', 'Antihypertensive', 'Antiplatelets', 'Anticoagulant',
                          'Antidiabetic', 'Thyroidal Medication', 'Antipsychotic', 'Betablocker',
                          'Cholesterol Lowering Drug', 'Diuratic', 'OtherMedication']
        medication_count = 0
        for col in medication_cols:
            if col in features_df.columns:
                val = features_df[col].iloc[0]
                if isinstance(val, str):
                    if val.lower() in ['yes', 'true', '1']:
                        medication_count += 1
                elif val == 1 or val is True:
                    medication_count += 1
        features_df['medication_count'] = medication_count
    
    # Calculate BMI if not provided but Weight and Height are
    if ('BMI' not in features_df.columns or pd.isna(features_df['BMI'].iloc[0])) and 'Weight' in features_df.columns and 'Height' in features_df.columns:
        weight = features_df['Weight'].iloc[0]
        height = features_df['Height'].iloc[0]
        if not pd.isna(weight) and not pd.isna(height) and height > 0:
            features_df['BMI'] = weight / ((height/100) ** 2)
        else:
            features_df['BMI'] = df_template['BMI'].median() if 'BMI' in df_template.columns else 25.0
    
    # Calculate BMI_Category if not provided
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
    
    # Calculate days_admission_to_surgery if not provided
    if 'days_admission_to_surgery' not in features_df.columns or pd.isna(features_df['days_admission_to_surgery'].iloc[0]):
        if 'Admission Date' in features_df.columns and 'Date of Surgery' in features_df.columns:
            try:
                admission = pd.to_datetime(features_df['Admission Date'].iloc[0])
                surgery = pd.to_datetime(features_df['Date of Surgery'].iloc[0])
                features_df['days_admission_to_surgery'] = (surgery - admission).days
            except:
                features_df['days_admission_to_surgery'] = df_template['days_admission_to_surgery'].median() if 'days_admission_to_surgery' in df_template.columns else 1
        else:
            features_df['days_admission_to_surgery'] = df_template['days_admission_to_surgery'].median() if 'days_admission_to_surgery' in df_template.columns else 1
    
    # CRITICAL: Use feature_columns (what model expects), NOT selected_features
    # The model was trained on feature_columns, so we must use those exact features
    if feature_columns:
        features_to_use = feature_columns.copy()
    else:
        # Fallback: try to load VIF-selected features
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent / 'config'))
            from selected_features_vif import SELECTED_FEATURES_VIF
            features_to_use = SELECTED_FEATURES_VIF.copy()
            if 'Complication Post Surgery' in features_to_use:
                features_to_use.remove('Complication Post Surgery')
        except ImportError:
            # Last resort: use template columns
            features_to_use = [col for col in df_template.columns if col != 'Complication Post Surgery']
    
    # Create features DataFrame with only the features the model expects
    # CRITICAL: Preserve user-provided values exactly as provided
    X = pd.DataFrame(index=[0])
    for col in features_to_use:
        # CRITICAL: Check if user provided this value
        if col in features_df.columns:
            val = features_df[col].iloc[0]
            # If user provided and value is not NaN/empty, USE IT
            if col in user_provided:
                if pd.isna(val) or (isinstance(val, str) and val.lower() in ['', 'nan', 'none', 'unknown']):
                    # User provided but it's empty/NaN - will fill with default below
                    X[col] = np.nan
                else:
                    # USER PROVIDED VALUE - USE IT!
                    X[col] = val
            else:
                # Not provided by user - use what's in features_df (might be calculated like BMI)
                X[col] = val
        else:
            # Feature not in input - will be filled with default below
            X[col] = np.nan
    
    # Handle missing values (same as training) - Fill ONLY truly missing values
    # CRITICAL: Only fill if value is actually NaN, preserve all user-provided values
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col in X.columns and X[col].isnull().any():
            # Check if this was a user-provided value that's NaN (shouldn't happen, but be safe)
            # Only fill if it's truly missing (not provided by user or calculated)
            if col not in user_provided or pd.isna(X[col].iloc[0]):
                # Fill with median from template
                if col in df_template.columns and df_template[col].dtype in [np.number]:
                    median_val = df_template[col].median()
                    X[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
                else:
                    X[col].fillna(0, inplace=True)
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in X.columns and X[col].isnull().any():
            # Only fill if it's truly missing
            if col not in user_provided or pd.isna(X[col].iloc[0]) or str(X[col].iloc[0]).lower() in ['', 'nan', 'none', 'unknown']:
                # Fill with mode from template
                if col in df_template.columns:
                    mode_val = df_template[col].mode()[0] if not df_template[col].mode().empty else 'Unknown'
                    X[col].fillna(mode_val, inplace=True)
                else:
                    X[col].fillna('Unknown', inplace=True)
    
    # Encode categorical variables (same as training)
    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col in X_encoded.columns:
            if encoders and col in encoders:
                # Use saved encoder
                try:
                    input_val = str(X_encoded[col].iloc[0])
                    if input_val in encoders[col].classes_:
                        X_encoded[col] = encoders[col].transform([input_val])[0]
                    else:
                        # Unseen value - use most common (usually index 0)
                        X_encoded[col] = 0
                except Exception as e:
                    X_encoded[col] = 0
            else:
                # Create new encoder from template
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
    
    # Handle datetime columns
    datetime_cols = X_encoded.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        for col in datetime_cols:
            try:
                min_date = df_template[col].min()
                X_encoded[col] = (pd.to_datetime(X_encoded[col]) - min_date).dt.days
            except:
                X_encoded = X_encoded.drop(columns=[col])
    
    # Remove zero-variance columns - BUT DON'T REMOVE USER-PROVIDED FEATURES!
    numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
    zero_var_cols = []
    for col in numeric_cols:
        if col in X_encoded.columns:
            # Don't remove if user provided this value
            if col not in user_provided:
                if X_encoded[col].var() == 0 or X_encoded[col].nunique() <= 1:
                    zero_var_cols.append(col)
    
    if zero_var_cols:
        X_encoded = X_encoded.drop(columns=zero_var_cols)
    
    # Ensure all columns are numeric
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
    
    # CRITICAL: Reorder to match training feature order exactly
    if feature_columns:
        # Add any missing columns that model expects
        for col in feature_columns:
            if col not in X_encoded.columns:
                # Get default from template
                if col in df_template.columns:
                    if df_template[col].dtype in [np.number]:
                        default_val = df_template[col].median() if not pd.isna(df_template[col].median()) else 0
                    else:
                        mode_val = df_template[col].mode()[0] if not df_template[col].mode().empty else 'Unknown'
                        # Encode the default value
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
        
        # Ensure exact feature order matches training
        missing_cols = [col for col in feature_columns if col not in X_encoded.columns]
        if missing_cols:
            for col in missing_cols:
                X_encoded[col] = 0
        
        # CRITICAL: Reorder but preserve existing values
        # Create new DataFrame with correct order, preserving existing values
        X_reordered = pd.DataFrame(index=[0])
        for col in feature_columns:
            if col in X_encoded.columns:
                # PRESERVE the existing value - use .values[0] to be safe
                val = X_encoded[col].values[0] if hasattr(X_encoded[col], 'values') else X_encoded[col].iloc[0]
                X_reordered[col] = val
            else:
                # Shouldn't happen (we added missing cols above), but fill with 0
                X_reordered[col] = 0
        X_encoded = X_reordered
        
        # Final check: ensure exact number of features
        if len(X_encoded.columns) != len(feature_columns):
            st.error(f"Feature mismatch: Expected {len(feature_columns)} features, got {len(X_encoded.columns)}")
            st.error(f"Missing: {set(feature_columns) - set(X_encoded.columns)}")
            st.error(f"Extra: {set(X_encoded.columns) - set(feature_columns)}")
    
    # Final verification: ensure all columns are numeric
    non_numeric_final = X_encoded.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_final) > 0:
        for col in non_numeric_final:
            X_encoded[col] = 0
    
    # Convert all to float - PRESERVE VALUES
    try:
        X_encoded = X_encoded.astype(float)
    except Exception as e:
        # Force conversion column by column - preserve user-provided values
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
    
    # Scale if scaler is provided (for Logistic Regression)
    if scaler:
        try:
            features_array = scaler.transform(X_encoded)
        except Exception as e:
            st.error(f"Scaling error: {e}")
            features_array = X_encoded.values.astype(float)
    else:
        features_array = X_encoded.values.astype(float)
    
    return features_array

@st.cache_resource
def load_risk_model():
    """
    Load the risk prediction model (VIF-selected features)
    Returns: model, scaler, encoders, feature_columns, metadata, error_message
    """
    risk_model_path = MODELS_DIR / 'best_risk_model.pkl'
    
    if not risk_model_path.exists():
        return None, None, None, None, None, "Risk prediction model not found. Please run scripts/risk_prediction_modeling.py first."
    
    try:
        model = joblib.load(risk_model_path)
        
        # Load scaler (may be None for XGBoost)
        scaler_path = MODELS_DIR / 'risk_scaler.pkl'
        scaler = None
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        
        # Load encoders
        encoders_path = MODELS_DIR / 'risk_encoders.pkl'
        encoders = None
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)
        
        # Load feature columns
        feature_columns_path = MODELS_DIR / 'risk_feature_columns.pkl'
        feature_columns = None
        if feature_columns_path.exists():
            feature_columns = joblib.load(feature_columns_path)
        
        # Load metadata
        metadata_path = MODELS_DIR / 'risk_model_metadata.pkl'
        metadata = None
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
        
        return model, scaler, encoders, feature_columns, metadata, None
        
    except Exception as e:
        return None, None, None, None, None, f"Error loading risk model: {str(e)}"

def get_risk_percentage(probability):
    """
    Convert model probability to risk percentage
    
    Args:
        probability: Model output probability (0-1 range)
    
    Returns:
        Risk percentage (0-100 range)
    """
    if probability is None or np.isnan(probability):
        return 0.0
    
    # Ensure probability is in valid range
    probability = max(0.0, min(1.0, float(probability)))
    
    # Convert to percentage
    risk_percentage = probability * 100.0
    
    return round(risk_percentage, 2)

def predict_risk(input_data, df_template):
    """
    Predict complication risk using the risk prediction model
    
    Args:
        input_data: Dictionary of input features
        df_template: DataFrame with the same structure as training data
    
    Returns:
        Dictionary with 'probability', 'risk_percentage', and 'prediction' (0 or 1)
    """
    # Load risk model
    model, scaler, encoders, feature_columns, metadata, error = load_risk_model()
    
    if error:
        return {
            'error': error,
            'probability': None,
            'risk_percentage': None,
            'prediction': None
        }
    
    # Prepare features
    try:
        features_array = prepare_features_for_prediction(
            input_data, df_template, scaler, encoders, feature_columns
        )
        
        # Make prediction
        probability = model.predict_proba(features_array)[0, 1]
        prediction = model.predict(features_array)[0]
        risk_percentage = get_risk_percentage(probability)
        
        return {
            'error': None,
            'probability': float(probability),
            'risk_percentage': risk_percentage,
            'prediction': int(prediction),
            'metadata': metadata
        }
        
    except Exception as e:
        return {
            'error': f"Prediction error: {str(e)}",
            'probability': None,
            'risk_percentage': None,
            'prediction': None
        }
