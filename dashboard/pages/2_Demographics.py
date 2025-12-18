"""
Demographics Page - Age, Gender, BMI, Geographic, Smoking Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.charts import histogram, box_plot, scatter_plot, bar_chart, pie_chart

st.title("Demographics Analysis")

# Get filtered data
if 'filtered_df' in st.session_state:
    df = st.session_state['filtered_df']
else:
    st.error("Please go to the main page first")
    st.stop()

# Age Analysis
st.header("Age Distribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Histogram")
    fig = px.histogram(
        df,
        x='Age',
        nbins=30,
        color='Gender',
        title="Age Distribution by Gender",
        barmode='overlay',
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Age Box Plot by Gender")
    fig = px.box(
        df,
        x='Gender',
        y='Age',
        title="Age Distribution by Gender",
        color='Gender'
    )
    st.plotly_chart(fig, use_container_width=True)

# Gender Breakdown
st.header("Gender Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gender Distribution")
    gender_counts = df['Gender'].value_counts()
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title="Gender Distribution"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Gender Statistics")
    gender_stats = df.groupby('Gender')['Age'].agg(['count', 'mean', 'std']).round(2)
    st.dataframe(gender_stats)

# Geographic Distribution
st.header("Geographic Distribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Governorate Distribution")
    gov_counts = df['Governorate'].value_counts()
    fig = px.bar(
        x=gov_counts.values,
        y=gov_counts.index,
        orientation='h',
        title="Patients by Governorate",
        labels={'x': 'Count', 'y': 'Governorate'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Insurance Type Distribution")
    if 'Insurance' in df.columns:
        ins_counts = df['Insurance'].value_counts()
        fig = px.pie(
            values=ins_counts.values,
            names=ins_counts.index,
            title="Insurance Type Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

# BMI Analysis
st.header("BMI Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("BMI Distribution")
    fig = px.histogram(
        df,
        x='BMI',
        nbins=30,
        color='Gender',
        title="BMI Distribution by Gender",
        barmode='overlay',
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("BMI Category by Gender")
    if 'BMI_Category' in df.columns:
        bmi_gender = pd.crosstab(df['BMI_Category'], df['Gender'])
        bmi_gender_melted = bmi_gender.reset_index().melt(
            id_vars='BMI_Category',
            var_name='Gender',
            value_name='Count'
        )
        fig = px.bar(
            bmi_gender_melted,
            x='BMI_Category',
            y='Count',
            color='Gender',
            title="BMI Category Distribution by Gender",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

# Weight vs Height
st.header("Anthropometric Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weight vs Height Scatter")
    fig = px.scatter(
        df,
        x='Height',
        y='Weight',
        color='Gender',
        title="Weight vs Height by Gender",
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Age vs BMI Scatter")
    fig = px.scatter(
        df,
        x='Age',
        y='BMI',
        color='Gender',
        title="Age vs BMI by Gender",
        trendline="ols"
    )
    st.plotly_chart(fig, use_container_width=True)

# Smoking Status
st.header("Smoking Status")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Smoking Status Distribution")
    if 'SmokingStatus' in df.columns:
        smoke_counts = df['SmokingStatus'].value_counts()
        fig = px.bar(
            x=smoke_counts.index,
            y=smoke_counts.values,
            title="Smoking Status Distribution",
            labels={'x': 'Smoking Status', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Smoking Status by Gender")
    if 'SmokingStatus' in df.columns:
        smoke_gender = pd.crosstab(df['SmokingStatus'], df['Gender'])
        smoke_gender_melted = smoke_gender.reset_index().melt(
            id_vars='SmokingStatus',
            var_name='Gender',
            value_name='Count'
        )
        fig = px.bar(
            smoke_gender_melted,
            x='SmokingStatus',
            y='Count',
            color='Gender',
            title="Smoking Status by Gender",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

# Insurance by Governorate Heatmap
st.header("Insurance by Governorate")
if 'Insurance' in df.columns and 'Governorate' in df.columns:
    ins_gov = pd.crosstab(df['Governorate'], df['Insurance'])
    fig = px.imshow(
        ins_gov,
        labels=dict(x="Insurance", y="Governorate", color="Count"),
        title="Insurance Type by Governorate",
        aspect="auto",
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)
