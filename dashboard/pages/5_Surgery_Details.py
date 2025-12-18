"""
Surgery Details Page - Procedure Types, Duration, Anesthesia
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Surgery Details Analysis")

# Get filtered data
if 'filtered_df' in st.session_state:
    df = st.session_state['filtered_df']
else:
    st.error("Please go to the main page first")
    st.stop()

# Surgery Type Distribution
st.header("Surgery Type Distribution")

if 'Surgery' in df.columns:
    surgery_counts = df['Surgery'].value_counts().head(20)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 20 Surgery Types")
        fig = px.bar(
            x=surgery_counts.values,
            y=surgery_counts.index,
            orientation='h',
            title="Top 20 Surgery Types",
            labels={'x': 'Count', 'y': 'Surgery Type'},
            color=surgery_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Surgery Type Statistics")
        st.metric("Total Unique Surgeries", df['Surgery'].nunique())
        st.metric("Most Common Surgery", surgery_counts.index[0] if len(surgery_counts) > 0 else "N/A")
        st.metric("Most Common Count", int(surgery_counts.values[0]) if len(surgery_counts) > 0 else 0)

# Emergency Status
st.header("Emergency Status Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Emergency vs Elective Ratio")
    if 'Emergency Status of surgery' in df.columns:
        emergency_counts = df['Emergency Status of surgery'].value_counts()
        fig = px.pie(
            values=emergency_counts.values,
            names=emergency_counts.index,
            title="Emergency vs Elective Surgeries"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Emergency Status Statistics")
    if 'Emergency Status of surgery' in df.columns:
        emergency_stats = df['Emergency Status of surgery'].value_counts()
        st.dataframe(emergency_stats)

# Surgery Duration
st.header("Surgery Duration Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Surgery Duration Distribution")
    if 'Duration Of Surgery' in df.columns:
        duration_df = df[df['Duration Of Surgery'].notna()]
        if len(duration_df) > 0:
            fig = px.histogram(
                duration_df,
                x='Duration Of Surgery',
                nbins=30,
                title="Surgery Duration Distribution",
                labels={'Duration Of Surgery': 'Duration (minutes)'}
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Surgery Duration by Emergency Status")
    if 'Duration Of Surgery' in df.columns and 'Emergency Status of surgery' in df.columns:
        duration_emergency_df = df[df['Duration Of Surgery'].notna()]
        if len(duration_emergency_df) > 0:
            fig = px.box(
                duration_emergency_df,
                x='Emergency Status of surgery',
                y='Duration Of Surgery',
                title="Surgery Duration by Emergency Status",
                color='Emergency Status of surgery'
            )
            st.plotly_chart(fig, use_container_width=True)

# Surgery Duration Statistics
if 'Duration Of Surgery' in df.columns:
    duration_values = df['Duration Of Surgery'].dropna()
    if len(duration_values) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Duration", f"{duration_values.mean():.1f} min")
        with col2:
            st.metric("Median Duration", f"{duration_values.median():.1f} min")
        with col3:
            st.metric("Min Duration", f"{duration_values.min():.1f} min")
        with col4:
            st.metric("Max Duration", f"{duration_values.max():.1f} min")

# Anesthesia Analysis
st.header("Anesthesia Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Anesthesia Type Distribution")
    if 'Anesthesia type' in df.columns:
        anesthesia_counts = df['Anesthesia type'].value_counts()
        fig = px.bar(
            x=anesthesia_counts.index,
            y=anesthesia_counts.values,
            title="Anesthesia Type Distribution",
            labels={'x': 'Anesthesia Type', 'y': 'Count'},
            color=anesthesia_counts.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Way of Anesthesia")
    if 'Way Of Anesthesia' in df.columns:
        way_counts = df['Way Of Anesthesia'].value_counts()
        fig = px.pie(
            values=way_counts.values,
            names=way_counts.index,
            title="Way of Anesthesia Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

# Surgery Duration by Anesthesia Type
if 'Duration Of Surgery' in df.columns and 'Anesthesia type' in df.columns:
    st.subheader("Surgery Duration by Anesthesia Type")
    duration_anesthesia_df = df[df['Duration Of Surgery'].notna()]
    if len(duration_anesthesia_df) > 0:
        fig = px.box(
            duration_anesthesia_df,
            x='Anesthesia type',
            y='Duration Of Surgery',
            title="Surgery Duration by Anesthesia Type",
            color='Anesthesia type'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Blood Transfusion
st.header("Blood Transfusion Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Blood Transfusion Rate")
    if 'Blood Transfusion During Surgery' in df.columns:
        transfusion_counts = df['Blood Transfusion During Surgery'].value_counts()
        fig = px.pie(
            values=transfusion_counts.values,
            names=transfusion_counts.index,
            title="Blood Transfusion During Surgery"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Number of Transfused Units")
    if 'Number of transfused PC during surgery' in df.columns:
        transfusion_units = df['Number of transfused PC during surgery'].dropna()
        if len(transfusion_units) > 0:
            fig = px.histogram(
                df,
                x='Number of transfused PC during surgery',
                nbins=10,
                title="Distribution of Transfused Units",
                labels={'Number of transfused PC during surgery': 'Number of Units'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Blood Transfusion by Surgery Duration
if 'Blood Transfusion During Surgery' in df.columns and 'Duration Of Surgery' in df.columns:
    st.subheader("Blood Transfusion by Surgery Duration")
    transfusion_duration_df = df[df['Duration Of Surgery'].notna()]
    if len(transfusion_duration_df) > 0:
        fig = px.scatter(
            transfusion_duration_df,
            x='Duration Of Surgery',
            y='Number of transfused PC during surgery',
            color='Blood Transfusion During Surgery',
            title="Blood Transfusion vs Surgery Duration",
            labels={'Duration Of Surgery': 'Duration (minutes)'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Days from Admission to Surgery
st.header("Time to Surgery")

if 'days_admission_to_surgery' in df.columns:
    days_to_surgery = df['days_admission_to_surgery'].dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Days from Admission to Surgery")
        if len(days_to_surgery) > 0:
            fig = px.histogram(
                df,
                x='days_admission_to_surgery',
                nbins=20,
                title="Days from Admission to Surgery",
                labels={'days_admission_to_surgery': 'Days'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Time to Surgery Statistics")
        if len(days_to_surgery) > 0:
            st.metric("Mean Days", f"{days_to_surgery.mean():.1f}")
            st.metric("Median Days", f"{days_to_surgery.median():.1f}")
            st.metric("Min Days", f"{days_to_surgery.min():.0f}")
            st.metric("Max Days", f"{days_to_surgery.max():.0f}")

# Complication During Surgery
st.header("Intraoperative Complications")

if 'Complication During Surgery' in df.columns:
    complication_counts = df['Complication During Surgery'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Complication During Surgery Rate")
        fig = px.pie(
            values=complication_counts.values,
            names=complication_counts.index,
            title="Complication During Surgery"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Complication by Emergency Status")
        if 'Emergency Status of surgery' in df.columns:
            comp_emergency = pd.crosstab(
                df['Complication During Surgery'],
                df['Emergency Status of surgery']
            )
            comp_emergency_melted = comp_emergency.reset_index().melt(
                id_vars='Complication During Surgery',
                var_name='Emergency Status',
                value_name='Count'
            )
            fig = px.bar(
                comp_emergency_melted,
                x='Complication During Surgery',
                y='Count',
                color='Emergency Status',
                title="Complication During Surgery by Emergency Status",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
