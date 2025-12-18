"""
Discharge and Follow-up Page - Hospitalization Duration and Outcomes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("Discharge and Follow-up Analysis")

# Get filtered data
if 'filtered_df' in st.session_state:
    df = st.session_state['filtered_df']
else:
    st.error("Please go to the main page first")
    st.stop()

# Hospitalization Duration
st.header("Hospitalization Duration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hospitalization Duration Distribution")
    if 'Duration of hospitalization (days)' in df.columns:
        hosp_duration = df['Duration of hospitalization (days)'].dropna()
        if len(hosp_duration) > 0:
            fig = px.histogram(
                df,
                x='Duration of hospitalization (days)',
                nbins=30,
                title="Hospitalization Duration Distribution",
                labels={'Duration of hospitalization (days)': 'Days'}
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hospitalization Duration Box Plot")
    if 'Duration of hospitalization (days)' in df.columns:
        hosp_duration = df['Duration of hospitalization (days)'].dropna()
        if len(hosp_duration) > 0:
            fig = px.box(
                df,
                y='Duration of hospitalization (days)',
                title="Hospitalization Duration Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# Hospitalization Duration Statistics
if 'Duration of hospitalization (days)' in df.columns:
    hosp_duration = df['Duration of hospitalization (days)'].dropna()
    if len(hosp_duration) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Stay", f"{hosp_duration.mean():.1f} days")
        with col2:
            st.metric("Median Stay", f"{hosp_duration.median():.1f} days")
        with col3:
            st.metric("Min Stay", f"{hosp_duration.min():.0f} days")
        with col4:
            st.metric("Max Stay", f"{hosp_duration.max():.0f} days")

# Hospitalization Duration by Complication Status
st.subheader("Hospitalization Duration by Complication Status")
if 'Duration of hospitalization (days)' in df.columns and 'Complication Post Surgery' in df.columns:
    hosp_comp_df = df[df['Duration of hospitalization (days)'].notna()]
    if len(hosp_comp_df) > 0:
        fig = px.box(
            hosp_comp_df,
            x='Complication Post Surgery',
            y='Duration of hospitalization (days)',
            title="Hospitalization Duration by Complication Status",
            color='Complication Post Surgery'
        )
        st.plotly_chart(fig, use_container_width=True)

# In-Hospital Mortality
st.header("In-Hospital Mortality")

col1, col2 = st.columns(2)

with col1:
    if 'Death post surgery during hospitalization' in df.columns:
        in_hosp_deaths = (df['Death post surgery during hospitalization'].astype(str).str.lower() == 'yes').sum()
        in_hosp_mortality_rate = (in_hosp_deaths / len(df)) * 100
        
        st.metric("In-Hospital Deaths", in_hosp_deaths)
        st.metric("In-Hospital Mortality Rate", f"{in_hosp_mortality_rate:.2f}%")

with col2:
    if 'Death post surgery during hospitalization' in df.columns:
        death_counts = df['Death post surgery during hospitalization'].value_counts()
        fig = px.pie(
            values=death_counts.values,
            names=death_counts.index,
            title="In-Hospital Mortality"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

# Post-Discharge Outcomes Summary
st.header("Post-Discharge Outcomes Summary")

# Filter for patients with follow-up
if 'follow_up_available' in df.columns:
    followup_df = df[df['follow_up_available'] == 1]
    st.info(f"Analysis based on {len(followup_df)} patients with follow-up data (out of {len(df)} total)")
    
    outcome_cols = [
        'Complication post Discharge', 'ER Visit', 'Readmission due to OR',
        'Infection or inflammation', 'Redo surgery', 'Admission into other hospital',
        'Death post discharge'
    ]
    
    outcome_data = []
    for col in outcome_cols:
        if col in followup_df.columns:
            count = (followup_df[col].astype(str).str.lower() == 'yes').sum()
            pct = (count / len(followup_df)) * 100 if len(followup_df) > 0 else 0
            outcome_data.append({
                'Outcome': col,
                'Count': count,
                'Percentage': pct
            })
    
    if outcome_data:
        outcome_df = pd.DataFrame(outcome_data).sort_values('Count', ascending=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Post-Discharge Outcomes (Count)")
            fig = px.bar(
                outcome_df,
                x='Count',
                y='Outcome',
                orientation='h',
                title="Post-Discharge Outcomes",
                color='Count',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Post-Discharge Outcomes (Percentage)")
            fig = px.bar(
                outcome_df,
                x='Percentage',
                y='Outcome',
                orientation='h',
                title="Post-Discharge Outcomes (%)",
                color='Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Follow-up Metrics
st.header("Follow-up Metrics")

if 'follow_up_available' in df.columns:
    followup_df = df[df['follow_up_available'] == 1]
    
    if len(followup_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            post_discharge_comp = (followup_df['Complication post Discharge'].astype(str).str.lower() == 'yes').sum() if 'Complication post Discharge' in followup_df.columns else 0
            st.metric("Post-Discharge Complications", post_discharge_comp)
        
        with col2:
            er_visits = (followup_df['ER Visit'].astype(str).str.lower() == 'yes').sum() if 'ER Visit' in followup_df.columns else 0
            st.metric("ER Visits", er_visits)
        
        with col3:
            readmissions = (followup_df['Readmission due to OR'].astype(str).str.lower() == 'yes').sum() if 'Readmission due to OR' in followup_df.columns else 0
            readmission_rate = (readmissions / len(followup_df)) * 100 if len(followup_df) > 0 else 0
            st.metric("Readmission Rate", f"{readmission_rate:.1f}%")
        
        with col4:
            post_discharge_deaths = (followup_df['Death post discharge'].astype(str).str.lower() == 'yes').sum() if 'Death post discharge' in followup_df.columns else 0
            post_discharge_mortality = (post_discharge_deaths / len(followup_df)) * 100 if len(followup_df) > 0 else 0
            st.metric("Post-Discharge Mortality", f"{post_discharge_mortality:.1f}%")

# Predictors Analysis
st.header("Predictors of Hospitalization Duration")

# Add age groups
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 30, 50, 70, 100],
    labels=['<30', '30-50', '50-70', '70+']
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hospitalization Duration by Comorbidity Count")
    if 'Duration of hospitalization (days)' in df.columns and 'comorbidity_count' in df.columns:
        hosp_comorb_df = df[df['Duration of hospitalization (days)'].notna()]
        if len(hosp_comorb_df) > 0:
            fig = px.box(
                hosp_comorb_df,
                x='comorbidity_count',
                y='Duration of hospitalization (days)',
                title="Hospitalization Duration by Comorbidity Count",
                labels={'comorbidity_count': 'Number of Comorbidities'}
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hospitalization Duration by Age Group")
    if 'Duration of hospitalization (days)' in df.columns and 'Age_Group' in df.columns:
        hosp_age_df = df[df['Duration of hospitalization (days)'].notna()]
        if len(hosp_age_df) > 0:
            fig = px.box(
                hosp_age_df,
                x='Age_Group',
                y='Duration of hospitalization (days)',
                title="Hospitalization Duration by Age Group",
                color='Age_Group'
            )
            st.plotly_chart(fig, use_container_width=True)

# Mortality by Comorbidity Count
st.subheader("Mortality by Comorbidity Count")
if 'comorbidity_count' in df.columns:
    if 'Death post surgery during hospitalization' in df.columns:
        mortality_comorb_data = []
        for comorb_count in sorted(df['comorbidity_count'].unique()):
            comorb_df = df[df['comorbidity_count'] == comorb_count]
            deaths = (comorb_df['Death post surgery during hospitalization'].astype(str).str.lower() == 'yes').sum()
            mortality_rate = (deaths / len(comorb_df)) * 100 if len(comorb_df) > 0 else 0
            mortality_comorb_data.append({
                'Comorbidity_Count': comorb_count,
                'Mortality_Rate': mortality_rate
            })
        
        if mortality_comorb_data:
            mortality_comorb_df = pd.DataFrame(mortality_comorb_data)
            fig = px.bar(
                mortality_comorb_df,
                x='Comorbidity_Count',
                y='Mortality_Rate',
                title="In-Hospital Mortality Rate by Comorbidity Count",
                labels={'Mortality_Rate': 'Mortality Rate (%)', 'Comorbidity_Count': 'Number of Comorbidities'},
                color='Mortality_Rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
