"""
Medical History Page - Comorbidities and Medications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.charts import bar_chart, histogram

st.title("Medical History Analysis")

# Get filtered data
if 'filtered_df' in st.session_state:
    df = st.session_state['filtered_df']
else:
    st.error("Please go to the main page first")
    st.stop()

# Add age groups
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 30, 50, 70, 100],
    labels=['<30', '30-50', '50-70', '70+']
)

# Comorbidity Prevalence
st.header("Comorbidity Prevalence")

comorbidity_cols = [
    'Hypertension', 'DiabetesMellitus', 'Dyslipidemia', 'CAD History',
    'HF', 'Open heart surgery', 'AFib-tachycardia', 'PAD', 'COPD', 'CKD',
    'Dialysis', 'Neurological/ Psychological disease', 'Gastrointestinal Disease',
    'Endocrine Disease', 'Cancer', 'Allergy'
]

comorbidity_data = []
for col in comorbidity_cols:
    if col in df.columns:
        count = (df[col].astype(str).str.lower() == 'yes').sum()
        pct = (count / len(df)) * 100
        comorbidity_data.append({
            'Condition': col,
            'Count': count,
            'Percentage': pct
        })

if comorbidity_data:
    comorb_df = pd.DataFrame(comorbidity_data).sort_values('Count', ascending=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Comorbidity Prevalence (Count)")
        fig = px.bar(
            comorb_df,
            x='Count',
            y='Condition',
            orientation='h',
            title="Comorbidity Prevalence",
            color='Count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Comorbidity Prevalence (Percentage)")
        fig = px.bar(
            comorb_df,
            x='Percentage',
            y='Condition',
            orientation='h',
            title="Comorbidity Prevalence (%)",
            color='Percentage',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Comorbidity Count Distribution
st.header("Comorbidity Count Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Comorbidity Count Distribution")
    if 'comorbidity_count' in df.columns:
        fig = px.histogram(
            df,
            x='comorbidity_count',
            nbins=15,
            title="Distribution of Comorbidity Count",
            labels={'comorbidity_count': 'Number of Comorbidities', 'count': 'Number of Patients'}
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Comorbidity Count Statistics")
    if 'comorbidity_count' in df.columns:
        stats = df['comorbidity_count'].describe()
        st.dataframe(stats)
        st.metric("Mean Comorbidities", f"{df['comorbidity_count'].mean():.2f}")
        st.metric("Median Comorbidities", f"{df['comorbidity_count'].median():.2f}")

# Comorbidity by Age Group
st.header("Comorbidity by Age Group")

if 'Age_Group' in df.columns:
    top_comorbs = ['Hypertension', 'DiabetesMellitus', 'Dyslipidemia', 'CAD History', 'CKD']
    comorb_age_data = []
    
    for comorb in top_comorbs:
        if comorb in df.columns:
            for age_group in df['Age_Group'].dropna().unique():
                age_df = df[df['Age_Group'] == age_group]
                count = (age_df[comorb].astype(str).str.lower() == 'yes').sum()
                total = len(age_df)
                pct = (count / total * 100) if total > 0 else 0
                comorb_age_data.append({
                    'Condition': comorb,
                    'Age_Group': str(age_group),
                    'Percentage': pct
                })
    
    if comorb_age_data:
        comorb_age_df = pd.DataFrame(comorb_age_data)
        fig = px.bar(
            comorb_age_df,
            x='Age_Group',
            y='Percentage',
            color='Condition',
            title="Comorbidity Prevalence by Age Group",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

# Comorbidity by Gender
st.header("Comorbidity by Gender")

top_comorbs_gender = ['Hypertension', 'DiabetesMellitus', 'Dyslipidemia', 'CAD History']
comorb_gender_data = []

for comorb in top_comorbs_gender:
    if comorb in df.columns:
        for gender in df['Gender'].dropna().unique():
            gender_df = df[df['Gender'] == gender]
            count = (gender_df[comorb].astype(str).str.lower() == 'yes').sum()
            total = len(gender_df)
            pct = (count / total * 100) if total > 0 else 0
            comorb_gender_data.append({
                'Condition': comorb,
                'Gender': gender,
                'Percentage': pct
            })

if comorb_gender_data:
    comorb_gender_df = pd.DataFrame(comorb_gender_data)
    fig = px.bar(
        comorb_gender_df,
        x='Condition',
        y='Percentage',
        color='Gender',
        title="Comorbidity Prevalence by Gender",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

# Medication Analysis
st.header("Medication Analysis")

medication_cols = [
    'Antihypertensive', 'Antiplatelets', 'Anticoagulant', 'Antidiabetic',
    'Thyroidal Medication', 'Antipsychotic', 'Betablocker',
    'Cholesterol Lowering Drug', 'Diuratic'
]

medication_data = []
for col in medication_cols:
    if col in df.columns:
        count = (df[col].astype(str).str.lower() == 'yes').sum()
        medication_data.append({
            'Medication': col,
            'Count': count
        })

if medication_data:
    med_df = pd.DataFrame(medication_data).sort_values('Count', ascending=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Medication Usage")
        fig = px.bar(
            med_df,
            x='Count',
            y='Medication',
            orientation='h',
            title="Medication Usage",
            color='Count',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Medication Count Distribution")
        if 'medication_count' in df.columns:
            fig = px.histogram(
                df,
                x='medication_count',
                nbins=15,
                title="Distribution of Medication Count",
                labels={'medication_count': 'Number of Medications'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Diabetes vs Antidiabetic Correlation
st.header("Clinical Correlations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Diabetes vs Antidiabetic Usage")
    if 'DiabetesMellitus' in df.columns and 'Antidiabetic' in df.columns:
        diabetes_yes = df[df['DiabetesMellitus'].astype(str).str.lower() == 'yes']
        diabetes_no = df[df['DiabetesMellitus'].astype(str).str.lower() == 'no']
        
        antidiab_yes_diabetes = (diabetes_yes['Antidiabetic'].astype(str).str.lower() == 'yes').sum()
        antidiab_no_diabetes = (diabetes_no['Antidiabetic'].astype(str).str.lower() == 'yes').sum()
        
        correlation_data = pd.DataFrame({
            'Diabetes Status': ['Has Diabetes', 'No Diabetes'],
            'On Antidiabetic': [antidiab_yes_diabetes, antidiab_no_diabetes],
            'Not on Antidiabetic': [
                len(diabetes_yes) - antidiab_yes_diabetes,
                len(diabetes_no) - antidiab_no_diabetes
            ]
        })
        
        fig = px.bar(
            correlation_data,
            x='Diabetes Status',
            y=['On Antidiabetic', 'Not on Antidiabetic'],
            title="Antidiabetic Usage by Diabetes Status",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Comorbidity vs Medication Count")
    if 'comorbidity_count' in df.columns and 'medication_count' in df.columns:
        fig = px.scatter(
            df,
            x='comorbidity_count',
            y='medication_count',
            color='Gender',
            title="Comorbidity Count vs Medication Count",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
