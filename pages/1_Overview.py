"""
Overview Page - Key Metrics and Summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dashboard.utils.charts import pie_chart, bar_chart, line_chart

st.title("Overview Dashboard")

# Get filtered data
if 'filtered_df' in st.session_state:
    df = st.session_state['filtered_df']
else:
    st.error("Please go to the main page first")
    st.stop()

# Key Metrics Row
st.header("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Patients", len(df))

with col2:
    avg_age = df['Age'].mean()
    st.metric("Average Age", f"{avg_age:.1f} years")

with col3:
    avg_hosp = df['Duration of hospitalization (days)'].mean()
    st.metric("Avg Hospital Stay", f"{avg_hosp:.1f} days")

with col4:
    complication_rate = (df['Complication Post Surgery'].astype(str).str.lower() == 'yes').sum() / len(df) * 100
    st.metric("Complication Rate", f"{complication_rate:.1f}%")

with col5:
    in_hosp_deaths = (df['Death post surgery during hospitalization'].astype(str).str.lower() == 'yes').sum()
    post_discharge_deaths = (df['Death post discharge'].astype(str).str.lower() == 'yes').sum()
    total_deaths = in_hosp_deaths + post_discharge_deaths
    mortality_rate = (total_deaths / len(df)) * 100
    st.metric("Mortality Rate", f"{mortality_rate:.1f}%")

st.markdown("---")

# Summary Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Gender Distribution")
    gender_counts = df['Gender'].value_counts()
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Monthly Admission Trend")
    df['Admission_Date'] = pd.to_datetime(df['Admission Date'])
    df['Year_Month'] = df['Admission_Date'].dt.to_period('M').astype(str)
    monthly_counts = df.groupby('Year_Month').size().reset_index(name='Count')
    monthly_counts = monthly_counts.sort_values('Year_Month')
    
    fig = px.line(
        monthly_counts,
        x='Year_Month',
        y='Count',
        markers=True,
        title="Admissions Over Time"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Number of Admissions")
    st.plotly_chart(fig, use_container_width=True)

# Comorbidity Distribution
st.subheader("Top Comorbidities")
comorbidity_cols = ['Hypertension', 'DiabetesMellitus', 'Dyslipidemia', 'CAD History', 
                     'HF', 'COPD', 'CKD', 'Cancer']
comorbidity_data = []
for col in comorbidity_cols:
    if col in df.columns:
        count = (df[col].astype(str).str.lower() == 'yes').sum()
        comorbidity_data.append({'Condition': col, 'Count': count})

if comorbidity_data:
    comorb_df = pd.DataFrame(comorbidity_data).sort_values('Count', ascending=True)
    fig = px.bar(
        comorb_df,
        x='Count',
        y='Condition',
        orientation='h',
        title="Comorbidity Prevalence",
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Outcome Summary
st.subheader("Outcome Summary")
outcome_data = {
    'Outcome': ['Complications', 'Readmissions', 'Deaths'],
    'Count': [
        (df['Complication Post Surgery'].astype(str).str.lower() == 'yes').sum(),
        (df['Readmission due to OR'].astype(str).str.lower() == 'yes').sum() if 'Readmission due to OR' in df.columns else 0,
        total_deaths
    ]
}
outcome_df = pd.DataFrame(outcome_data)
fig = px.bar(
    outcome_df,
    x='Outcome',
    y='Count',
    title="Key Outcomes",
    color='Count',
    color_continuous_scale='Reds'
)
st.plotly_chart(fig, use_container_width=True)
