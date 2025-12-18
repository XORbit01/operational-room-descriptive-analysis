"""
Risk Prediction Page - Predict Complication Risk for New Patients
"""

import streamlit as st
import pandas as pd
import numpy as np
from dashboard.utils.data_loader import load_data
from dashboard.utils.model_loader import load_risk_model, predict_risk, get_risk_percentage
import plotly.graph_objects as go

st.title("Complication Risk Prediction")
st.markdown("Predict the risk of post-surgery complications for a new patient using our trained machine learning model.")

# Load data template
df_template = load_data()

# Load model and check if available
model, scaler, encoders, feature_columns, metadata, error = load_risk_model()

if error:
    st.error(f"**Model Loading Error:** {error}")
    st.info("""
    **To use the prediction feature:**
    1. Run feature selection: `python scripts/feature_selection_vif.py`
    2. Train the model: `python scripts/risk_prediction_modeling.py`
    3. Refresh this page
    """)
    st.stop()

# Display model info
if metadata:
    st.sidebar.info(f"""
    **Model Information:**
    - Model: {metadata.get('model_name', 'Unknown')}
    - ROC-AUC: {metadata.get('roc_auc', 0):.3f}
    - Features: {metadata.get('n_features', 0)}
    - Trained: {metadata.get('date_trained', 'Unknown')}
    """)

# ============================================================================
# Input Form
# ============================================================================

st.header("Patient Information")

# Create form for patient input
with st.form("prediction_form"):
    st.subheader("Demographics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)
        gender = st.selectbox("Gender", options=['male', 'female'], index=0)
    
    with col2:
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
        weight = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70, step=1)
    
    with col3:
        # Calculate BMI
        if height > 0:
            bmi = weight / ((height / 100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            if bmi < 18.5:
                bmi_category = 'underweight'
            elif bmi < 25:
                bmi_category = 'normal'
            elif bmi < 30:
                bmi_category = 'overweight'
            else:
                bmi_category = 'obese'
        else:
            bmi = 0
            bmi_category = 'normal'
    
    # Get unique values from template for categorical fields
    governorate_options = [''] + sorted(df_template['Governorate'].dropna().unique().tolist()) if 'Governorate' in df_template.columns else ['']
    marital_status_options = [''] + sorted(df_template['MaritalStatus'].dropna().unique().tolist()) if 'MaritalStatus' in df_template.columns else ['']
    insurance_options = [''] + sorted(df_template['Insurance'].dropna().unique().tolist()) if 'Insurance' in df_template.columns else ['']
    smoking_options = [''] + sorted(df_template['SmokingStatus'].dropna().unique().tolist()) if 'SmokingStatus' in df_template.columns else ['']
    blood_group_options = [''] + sorted(df_template['BloodGroup'].dropna().unique().tolist()) if 'BloodGroup' in df_template.columns else ['']
    
    st.subheader("Additional Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        governorate = st.selectbox("Governorate", options=governorate_options, index=0)
        marital_status = st.selectbox("Marital Status", options=marital_status_options, index=0)
    with col2:
        insurance = st.selectbox("Insurance", options=insurance_options, index=0)
        smoking_status = st.selectbox("Smoking Status", options=smoking_options, index=0)
    with col3:
        blood_group = st.selectbox("Blood Group", options=blood_group_options, index=0)
    
    st.subheader("Pre-Surgery Lab Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        pre_bun = st.number_input("Pre-BUN", min_value=0.0, max_value=100.0, value=15.0, step=0.1, format="%.1f")
        pre_na = st.number_input("Pre Na (mmol/L)", min_value=100.0, max_value=160.0, value=140.0, step=0.1, format="%.1f")
    with col2:
        pre_hb = st.number_input("Pre HB (g/dL)", min_value=5.0, max_value=20.0, value=12.0, step=0.1, format="%.1f")
        pre_platelet = st.number_input("Pre Platelet (×10³/μL)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0, format="%.0f")
    with col3:
        pre_creatinine = st.number_input("Pre-Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.2f")
    
    st.subheader("Medical History & Comorbidities")
    col1, col2 = st.columns(2)
    
    with col1:
        hypertension = st.selectbox("Hypertension", options=['no', 'yes'], index=0)
        diabetes = st.selectbox("Diabetes Mellitus", options=['no', 'yes'], index=0)
        dyslipidemia = st.selectbox("Dyslipidemia", options=['no', 'yes'], index=0)
        cad_history = st.selectbox("CAD History", options=['no', 'yes'], index=0)
        hf = st.selectbox("Heart Failure (HF)", options=['no', 'yes'], index=0)
        open_heart_surgery = st.selectbox("Open Heart Surgery", options=['no', 'yes'], index=0)
        afib = st.selectbox("AFib/Tachycardia", options=['no', 'yes'], index=0)
        pad = st.selectbox("PAD", options=['no', 'yes'], index=0)
    
    with col2:
        copd = st.selectbox("COPD", options=['no', 'yes'], index=0)
        ckd = st.selectbox("CKD", options=['no', 'yes'], index=0)
        dialysis = st.selectbox("Dialysis", options=['no', 'yes'], index=0)
        neuro_psych = st.selectbox("Neurological/Psychological Disease", options=['no', 'yes'], index=0)
        gi_disease = st.selectbox("Gastrointestinal Disease", options=['no', 'yes'], index=0)
        endocrine = st.selectbox("Endocrine Disease", options=['no', 'yes'], index=0)
        cancer = st.selectbox("Cancer", options=['no', 'yes'], index=0)
        allergy = st.selectbox("Allergy", options=['no', 'yes'], index=0)
    
    st.subheader("Surgical Planning")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        emergency_status = st.selectbox("Emergency Status", 
                                       options=[''] + sorted(df_template['Emergency Status of surgery'].dropna().unique().tolist()) if 'Emergency Status of surgery' in df_template.columns else [''],
                                       index=0)
    with col2:
        anesthesia_type = st.selectbox("Anesthesia Type",
                                       options=[''] + sorted(df_template['Anesthesia type'].dropna().unique().tolist()) if 'Anesthesia type' in df_template.columns else [''],
                                       index=0)
    with col3:
        way_of_anesthesia = st.selectbox("Way of Anesthesia",
                                         options=[''] + sorted(df_template['Way Of Anesthesia'].dropna().unique().tolist()) if 'Way Of Anesthesia' in df_template.columns else [''],
                                         index=0)
    
    # Calculate comorbidity count
    comorbidity_list = [hypertension, diabetes, dyslipidemia, cad_history, hf, 
                       open_heart_surgery, afib, pad, copd, ckd, dialysis, 
                       neuro_psych, gi_disease, endocrine, cancer, allergy]
    comorbidity_count = sum(1 for c in comorbidity_list if c == 'yes')
    
    # Medication information (simplified - can be expanded)
    st.subheader("Current Medications")
    current_medication = st.selectbox("Current Medication", options=['no', 'yes'], index=0)
    
    # Additional surgical details
    st.subheader("Surgical Details")
    col1, col2 = st.columns(2)
    with col1:
        duration_surgery = st.number_input("Duration of Surgery (minutes)", min_value=0, max_value=1000, value=120, step=5)
        blood_transfusion = st.selectbox("Blood Transfusion During Surgery", options=['no', 'yes'], index=0)
    with col2:
        # Calculate days admission to surgery (simplified)
        days_admission_to_surgery = st.number_input("Days from Admission to Surgery", min_value=0, max_value=365, value=1, step=1)
    
    # Submit button
    submitted = st.form_submit_button("Predict Complication Risk", use_container_width=True)

# ============================================================================
# Prediction and Results
# ============================================================================

if submitted:
    st.markdown("---")
    st.header("Prediction Results")
    
    # Prepare input data dictionary
    input_data = {
        'Age': age,
        'Gender': gender,
        'Height': height,
        'BMI': bmi,
        'BMI_Category': bmi_category,
        'Governorate': governorate if governorate else 'Unknown',
        'MaritalStatus': marital_status if marital_status else 'Unknown',
        'Insurance': insurance if insurance else 'Unknown',
        'SmokingStatus': smoking_status if smoking_status else 'Unknown',
        'BloodGroup': blood_group if blood_group else 'Unknown',
        'Pre-BUN': pre_bun,
        'Pre Na': pre_na,
        'Pre HB': pre_hb,
        'Pre Platelet': pre_platelet,
        'Pre-Creatinine': pre_creatinine,
        'Duration Of Surgery': duration_surgery,
        'comorbidity_count': comorbidity_count,
        'medication_count': 1 if current_medication == 'yes' else 0,
        'days_admission_to_surgery': days_admission_to_surgery,
        'Hypertension': hypertension,
        'DiabetesMellitus': diabetes,
        'Dyslipidemia': dyslipidemia,
        'CAD History': cad_history,
        'HF': hf,
        'Open heart surgery': open_heart_surgery,
        'AFib-tachycardia': afib,
        'PAD': pad,
        'COPD': copd,
        'CKD': ckd,
        'Dialysis': dialysis,
        'Neurological/ Psychological disease': neuro_psych,
        'Gastrointestinal Disease': gi_disease,
        'Endocrine Disease': endocrine,
        'Cancer': cancer,
        'Allergy': allergy,
        'Current Medication': current_medication,
        'Emergency Status of surgery': emergency_status if emergency_status else 'Unknown',
        'Anesthesia type': anesthesia_type if anesthesia_type else 'Unknown',
        'Way Of Anesthesia': way_of_anesthesia if way_of_anesthesia else 'Unknown',
        'Blood Transfusion During Surgery': blood_transfusion,
    }
    
    # DON'T add all template columns - let the model loader handle missing features
    # Only add essential engineered features if not calculated
    if 'comorbidity_count' not in input_data:
        # Already calculated above, but ensure it's in input_data
        input_data['comorbidity_count'] = comorbidity_count
    
    if 'medication_count' not in input_data:
        input_data['medication_count'] = 1 if current_medication == 'yes' else 0
    
    # Make prediction
    with st.spinner("Calculating risk prediction..."):
        result = predict_risk(input_data, df_template)
    
    if result['error']:
        st.error(f"**Prediction Error:** {result['error']}")
    else:
        # Display results
        risk_percentage = result['risk_percentage']
        probability = result['probability']
        prediction = result['prediction']
        
        # Risk level classification
        if risk_percentage < 30:
            risk_level = "Low Risk"
            risk_color = "green"
        elif risk_percentage < 60:
            risk_level = "Moderate Risk"
            risk_color = "orange"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        # Main result display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Complication Risk", f"{risk_percentage:.1f}%", delta=None)
        
        with col2:
            st.metric("Risk Level", risk_level)
        
        with col3:
            prediction_text = "Complication Likely" if prediction == 1 else "No Complication Expected"
            st.metric("Prediction", prediction_text)
        
        # Visual risk indicator
        st.markdown("### Risk Visualization")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Complication Risk (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed information
        with st.expander("Detailed Prediction Information"):
            st.write(f"**Probability:** {probability:.4f} (0-1 range)")
            st.write(f"**Risk Percentage:** {risk_percentage:.2f}%")
            st.write(f"**Binary Prediction:** {'Complication' if prediction == 1 else 'No Complication'}")
            
            if metadata:
                st.write(f"**Model Used:** {metadata.get('model_name', 'Unknown')}")
                st.write(f"**Model ROC-AUC:** {metadata.get('roc_auc', 0):.3f}")
        
        # Clinical interpretation
        st.markdown("### Clinical Interpretation")
        
        if risk_percentage < 30:
            st.success(f"""
            **Low Risk ({risk_percentage:.1f}%)**
            
            The patient has a low risk of post-surgery complications. Standard surgical protocols and monitoring are recommended.
            """)
        elif risk_percentage < 60:
            st.warning(f"""
            **Moderate Risk ({risk_percentage:.1f}%)**
            
            The patient has a moderate risk of complications. Consider:
            - Enhanced pre-operative optimization
            - Close post-operative monitoring
            - Prophylactic measures as indicated
            """)
        else:
            st.error(f"""
            **High Risk ({risk_percentage:.1f}%)**
            
            The patient has a high risk of complications. Strongly consider:
            - Comprehensive pre-operative assessment and optimization
            - Intensive post-operative monitoring
            - Prophylactic interventions
            - Multidisciplinary team involvement
            - Consideration of alternative treatment approaches if appropriate
            """)
        
        # Key risk factors
        st.markdown("### Key Risk Factors Identified")
        risk_factors = []
        if age >= 65:
            risk_factors.append(f"Advanced age ({age} years)")
        if bmi >= 30:
            risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
        if comorbidity_count >= 3:
            risk_factors.append(f"Multiple comorbidities ({comorbidity_count})")
        if ckd == 'yes' or dialysis == 'yes':
            risk_factors.append("Renal disease")
        if emergency_status and 'emergency' in str(emergency_status).lower():
            risk_factors.append("Emergency surgery")
        if pre_creatinine > 1.2:
            risk_factors.append(f"Elevated creatinine ({pre_creatinine:.2f} mg/dL)")
        if pre_hb < 12:
            risk_factors.append(f"Low hemoglobin ({pre_hb:.1f} g/dL)")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.info("No major risk factors identified from the provided information.")

# ============================================================================
# Model Information
# ============================================================================

st.markdown("---")
st.sidebar.header("About This Model")
st.sidebar.info("""
This risk prediction model uses:
- **VIF-selected features** to avoid redundancy
- **XGBoost or Logistic Regression** algorithms
- **Class imbalance handling** (SMOTE/Class Weights)
- **Pre-surgery features only** (no data leakage)

The model predicts the probability of post-surgery complications based on patient characteristics available before surgery.
""")

if metadata:
    st.sidebar.markdown("### Model Performance")
    st.sidebar.write(f"**ROC-AUC:** {metadata.get('roc_auc', 0):.3f}")
    st.sidebar.write(f"**Features:** {metadata.get('n_features', 0)}")
    st.sidebar.write(f"**Trained:** {metadata.get('date_trained', 'Unknown')}")
