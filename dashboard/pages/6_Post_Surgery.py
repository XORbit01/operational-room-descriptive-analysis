"""
Post-Surgery Outcomes Page - Complications, ICU, Lab Changes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.charts import gauge_chart

st.title("Post-Surgery Outcomes Analysis")

# Get filtered data
if 'filtered_df' in st.session_state:
    df = st.session_state['filtered_df']
else:
    st.error("Please go to the main page first")
    st.stop()

# Overall Complication Rate
st.header("Overall Complication Rate")

if 'Complication Post Surgery' in df.columns:
    complication_rate = (df['Complication Post Surgery'].astype(str).str.lower() == 'yes').sum() / len(df) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Complication Rate", f"{complication_rate:.1f}%")
        st.metric("Patients with Complications", 
                 (df['Complication Post Surgery'].astype(str).str.lower() == 'yes').sum())
        st.metric("Patients without Complications",
                 (df['Complication Post Surgery'].astype(str).str.lower() != 'yes').sum())
    
    with col2:
        fig = gauge_chart(complication_rate, 100, "Complication Rate (%)", threshold=30)
        st.plotly_chart(fig, use_container_width=True)

# Complication Types Breakdown
st.header("Complication Types Breakdown")

complication_cols = [
    'Cardiac Complication', 'Pulmonary complication', 'Renal complication',
    'Neurological complication', 'Stroke', 'Coma', 'Major wound disruption',
    'Infection of the surgical site', 'Bacteremia', 'Sepsis', 'Septic Shock'
]

complication_data = []
for col in complication_cols:
    if col in df.columns:
        count = (df[col].astype(str).str.lower() == 'yes').sum()
        if count > 0:
            complication_data.append({
                'Complication': col,
                'Count': count
            })

if complication_data:
    comp_df = pd.DataFrame(complication_data).sort_values('Count', ascending=True)
    
    fig = px.bar(
        comp_df,
        x='Count',
        y='Complication',
        orientation='h',
        title="Post-Surgery Complication Types",
        color='Count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Complication Count Distribution
st.header("Complication Count Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Complication Count Distribution")
    if 'complication_count' in df.columns:
        fig = px.histogram(
            df,
            x='complication_count',
            nbins=15,
            title="Distribution of Complication Count",
            labels={'complication_count': 'Number of Complications'}
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Complication Count Statistics")
    if 'complication_count' in df.columns:
        stats = df['complication_count'].describe()
        st.dataframe(stats)
        st.metric("Mean Complications", f"{df['complication_count'].mean():.2f}")
        st.metric("Patients with 0 Complications", 
                 (df['complication_count'] == 0).sum())

# ICU Transfer Analysis
st.header("ICU Transfer Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ICU Transfer Rate")
    if 'Unplanned transfer to intensive care unit' in df.columns:
        icu_counts = df['Unplanned transfer to intensive care unit'].value_counts()
        fig = px.pie(
            values=icu_counts.values,
            names=icu_counts.index,
            title="ICU Transfer Rate"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("ICU Duration Distribution")
    if 'Duration in intensive care unit (days)' in df.columns:
        icu_duration = df[df['Duration in intensive care unit (days)'].notna()]
        if len(icu_duration) > 0:
            fig = px.histogram(
                icu_duration,
                x='Duration in intensive care unit (days)',
                nbins=15,
                title="ICU Duration Distribution",
                labels={'Duration in intensive care unit (days)': 'Days in ICU'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Lab Value Changes
st.header("Lab Value Changes (Pre vs Post-Op)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pre vs Post-Op Hemoglobin")
    if 'Pre HB' in df.columns and 'HB day 1 post surgery' in df.columns:
        # Get patients with both values
        hb_complete = df[df['Pre HB'].notna() & df['HB day 1 post surgery'].notna()]
        if len(hb_complete) > 0:
            hb_data = []
            for idx, row in hb_complete.iterrows():
                hb_data.append({'Timepoint': 'Pre-Op', 'Hemoglobin': row['Pre HB']})
                hb_data.append({'Timepoint': 'Post-Op Day 1', 'Hemoglobin': row['HB day 1 post surgery']})
            
            hb_df = pd.DataFrame(hb_data)
            fig = px.box(
                hb_df,
                x='Timepoint',
                y='Hemoglobin',
                title="Hemoglobin: Pre vs Post-Op",
                color='Timepoint'
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Pre vs Post-Op Creatinine")
    if 'Pre-Creatinine' in df.columns and 'Creatinine_D1' in df.columns:
        # Get patients with both values
        creat_complete = df[df['Pre-Creatinine'].notna() & df['Creatinine_D1'].notna()]
        if len(creat_complete) > 0:
            creat_data = []
            for idx, row in creat_complete.iterrows():
                creat_data.append({'Timepoint': 'Pre-Op', 'Creatinine': row['Pre-Creatinine']})
                creat_data.append({'Timepoint': 'Post-Op Day 1', 'Creatinine': row['Creatinine_D1']})
            
            creat_df = pd.DataFrame(creat_data)
            fig = px.box(
                creat_df,
                x='Timepoint',
                y='Creatinine',
                title="Creatinine: Pre vs Post-Op",
                color='Timepoint'
            )
            st.plotly_chart(fig, use_container_width=True)

# Lab Deltas Distribution
st.header("Lab Value Deltas")

delta_cols = {
    'hb_delta_d1': 'Hemoglobin Delta (Day 1)',
    'creatinine_delta_d1': 'Creatinine Delta (Day 1)',
    'na_delta_d1': 'Sodium Delta (Day 1)'
}

delta_data = []
for col, label in delta_cols.items():
    if col in df.columns:
        delta_values = df[col].dropna()
        for val in delta_values:
            delta_data.append({'Delta': label, 'Value': val})

if delta_data:
    delta_df = pd.DataFrame(delta_data)
    fig = px.box(
        delta_df,
        x='Delta',
        y='Value',
        title="Lab Value Changes (Post-Op Day 1 - Pre-Op)",
        color='Delta'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Risk Analysis
st.header("Risk Analysis")

# Add age groups
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 30, 50, 70, 100],
    labels=['<30', '30-50', '50-70', '70+']
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Complications by Age Group")
    if 'Age_Group' in df.columns and 'Complication Post Surgery' in df.columns:
        comp_age_data = []
        for age_group in df['Age_Group'].dropna().unique():
            age_df = df[df['Age_Group'] == age_group]
            comp_rate = (age_df['Complication Post Surgery'].astype(str).str.lower() == 'yes').sum() / len(age_df) * 100
            comp_age_data.append({
                'Age_Group': str(age_group),
                'Complication_Rate': comp_rate
            })
        
        if comp_age_data:
            comp_age_df = pd.DataFrame(comp_age_data)
            fig = px.bar(
                comp_age_df,
                x='Age_Group',
                y='Complication_Rate',
                title="Complication Rate by Age Group",
                labels={'Complication_Rate': 'Complication Rate (%)', 'Age_Group': 'Age Group'},
                color='Complication_Rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Complications by Comorbidity Count")
    if 'comorbidity_count' in df.columns and 'Complication Post Surgery' in df.columns:
        comp_comorb_data = []
        for comorb_count in sorted(df['comorbidity_count'].unique()):
            comorb_df = df[df['comorbidity_count'] == comorb_count]
            comp_rate = (comorb_df['Complication Post Surgery'].astype(str).str.lower() == 'yes').sum() / len(comorb_df) * 100
            comp_comorb_data.append({
                'Comorbidity_Count': comorb_count,
                'Complication_Rate': comp_rate
            })
        
        if comp_comorb_data:
            comp_comorb_df = pd.DataFrame(comp_comorb_data)
            fig = px.bar(
                comp_comorb_df,
                x='Comorbidity_Count',
                y='Complication_Rate',
                title="Complication Rate by Comorbidity Count",
                labels={'Complication_Rate': 'Complication Rate (%)', 'Comorbidity_Count': 'Number of Comorbidities'},
                color='Complication_Rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)

# Complications by Emergency Status
st.subheader("Complications by Emergency Status")
if 'Emergency Status of surgery' in df.columns and 'Complication Post Surgery' in df.columns:
    comp_emergency = pd.crosstab(
        df['Complication Post Surgery'],
        df['Emergency Status of surgery']
    )
    # Reshape for Plotly
    comp_emergency_melted = comp_emergency.reset_index().melt(
        id_vars='Complication Post Surgery',
        var_name='Emergency Status',
        value_name='Count'
    )
    fig = px.bar(
        comp_emergency_melted,
        x='Complication Post Surgery',
        y='Count',
        color='Emergency Status',
        title="Complications by Emergency Status",
        barmode='group',
        labels={'Complication Post Surgery': 'Complication Status'}
    )
    st.plotly_chart(fig, use_container_width=True)
