"""
Pre-Surgery Analysis Page - Lab Values and Clinical Indicators
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("Pre-Surgery Analysis")

# Get filtered data
if 'filtered_df' in st.session_state:
    df = st.session_state['filtered_df']
else:
    st.error("Please go to the main page first")
    st.stop()

# Pre-Surgery Lab Distributions
st.header("Pre-Surgery Laboratory Values")

lab_cols = {
    'Pre-BUN': 'BUN (mg/dL)',
    'Pre-Creatinine': 'Creatinine (mg/dL)',
    'Pre Na': 'Sodium (mEq/L)',
    'Pre HB': 'Hemoglobin (g/dL)',
    'Pre Platelet': 'Platelet Count (x10³/µL)'
}

# Box plots for each lab
col1, col2 = st.columns(2)

with col1:
    st.subheader("Lab Values Distribution (Box Plot)")
    lab_data = []
    for col, label in lab_cols.items():
        if col in df.columns:
            lab_values = df[col].dropna()
            for val in lab_values:
                lab_data.append({'Lab': label, 'Value': val})
    
    if lab_data:
        lab_df = pd.DataFrame(lab_data)
        fig = px.box(
            lab_df,
            x='Lab',
            y='Value',
            title="Pre-Surgery Lab Values Distribution",
            color='Lab'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Lab Values by Gender")
    lab_gender_data = []
    for col, label in lab_cols.items():
        if col in df.columns:
            for gender in df['Gender'].dropna().unique():
                gender_df = df[df['Gender'] == gender]
                lab_values = gender_df[col].dropna()
                for val in lab_values:
                    lab_gender_data.append({
                        'Lab': label,
                        'Gender': gender,
                        'Value': val
                    })
    
    if lab_gender_data:
        lab_gender_df = pd.DataFrame(lab_gender_data)
        fig = px.box(
            lab_gender_df,
            x='Lab',
            y='Value',
            color='Gender',
            title="Pre-Surgery Lab Values by Gender"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Lab Values by Age Group
st.header("Lab Values by Age Group")

df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 30, 50, 70, 100],
    labels=['<30', '30-50', '50-70', '70+']
)

if 'Age_Group' in df.columns:
    lab_age_data = []
    for col, label in lab_cols.items():
        if col in df.columns:
            for age_group in df['Age_Group'].dropna().unique():
                age_df = df[df['Age_Group'] == age_group]
                lab_values = age_df[col].dropna()
                for val in lab_values:
                    lab_age_data.append({
                        'Lab': label,
                        'Age_Group': str(age_group),
                        'Value': val
                    })
    
    if lab_age_data:
        lab_age_df = pd.DataFrame(lab_age_data)
        fig = px.box(
            lab_age_df,
            x='Lab',
            y='Value',
            color='Age_Group',
            title="Pre-Surgery Lab Values by Age Group"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Clinical Indicators
st.header("Clinical Indicators")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Creatinine Distribution with CKD Highlighting")
    if 'Pre-Creatinine' in df.columns and 'CKD' in df.columns:
        ckd_patients = df[df['CKD'].astype(str).str.lower() == 'yes']
        non_ckd_patients = df[df['CKD'].astype(str).str.lower() != 'yes']
        
        fig = go.Figure()
        
        if len(non_ckd_patients) > 0:
            fig.add_trace(go.Histogram(
                x=non_ckd_patients['Pre-Creatinine'].dropna(),
                name='No CKD',
                opacity=0.7,
                marker_color='blue'
            ))
        
        if len(ckd_patients) > 0:
            fig.add_trace(go.Histogram(
                x=ckd_patients['Pre-Creatinine'].dropna(),
                name='CKD',
                opacity=0.7,
                marker_color='red'
            ))
        
        fig.update_layout(
            title="Creatinine Distribution by CKD Status",
            xaxis_title="Creatinine (mg/dL)",
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hemoglobin Distribution with Anemia Threshold")
    if 'Pre HB' in df.columns:
        fig = px.histogram(
            df,
            x='Pre HB',
            nbins=30,
            title="Hemoglobin Distribution",
            labels={'Pre HB': 'Hemoglobin (g/dL)'}
        )
        # Add anemia threshold line (typically <12 g/dL for women, <13 g/dL for men)
        fig.add_vline(x=12, line_dash="dash", line_color="red", annotation_text="Anemia Threshold")
        st.plotly_chart(fig, use_container_width=True)

# Lab Correlation Matrix
st.header("Lab Value Correlations")

lab_cols_for_corr = [col for col in lab_cols.keys() if col in df.columns]
if len(lab_cols_for_corr) > 1:
    lab_corr_df = df[lab_cols_for_corr].select_dtypes(include=[np.number])
    corr_matrix = lab_corr_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="Pre-Surgery Lab Values Correlation Matrix",
        aspect="auto",
        color_continuous_scale='RdBu',
        text_auto=True
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ER Admission Before Surgery
st.header("ER Admission and Imaging")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ER Admission Before Surgery")
    if 'ER Admission Before Surgery' in df.columns:
        er_counts = df['ER Admission Before Surgery'].value_counts()
        fig = px.pie(
            values=er_counts.values,
            names=er_counts.index,
            title="ER Admission Before Surgery"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Radiology/Imaging Usage")
    if 'Radiology' in df.columns:
        rad_counts = df['Radiology'].value_counts()
        fig = px.bar(
            x=rad_counts.index,
            y=rad_counts.values,
            title="Radiology/Imaging Usage",
            labels={'x': 'Radiology', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Lab Value Statistics Table
st.header("Lab Value Statistics")

lab_stats_data = []
for col, label in lab_cols.items():
    if col in df.columns:
        lab_values = df[col].dropna()
        if len(lab_values) > 0:
            lab_stats_data.append({
                'Lab Test': label,
                'Count': len(lab_values),
                'Mean': lab_values.mean(),
                'Median': lab_values.median(),
                'Std': lab_values.std(),
                'Min': lab_values.min(),
                'Max': lab_values.max()
            })

if lab_stats_data:
    lab_stats_df = pd.DataFrame(lab_stats_data)
    lab_stats_df = lab_stats_df.round(2)
    st.dataframe(lab_stats_df, use_container_width=True)
