"""
Data loading utilities for the dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Path to data file
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'data_cleaned.xlsx'

@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    df = pd.read_excel(DATA_PATH)
    return df

def apply_filters(df, filters):
    """Apply global filters to dataframe"""
    filtered_df = df.copy()
    
    if filters.get('gender'):
        filtered_df = filtered_df[filtered_df['Gender'].isin(filters['gender'])]
    
    if filters.get('age_range'):
        min_age, max_age = filters['age_range']
        filtered_df = filtered_df[(filtered_df['Age'] >= min_age) & (filtered_df['Age'] <= max_age)]
    
    if filters.get('governorate'):
        filtered_df = filtered_df[filtered_df['Governorate'].isin(filters['governorate'])]
    
    if filters.get('bmi_category'):
        filtered_df = filtered_df[filtered_df['BMI_Category'].isin(filters['bmi_category'])]
    
    if filters.get('emergency_status'):
        filtered_df = filtered_df[filtered_df['Emergency Status of surgery'].isin(filters['emergency_status'])]
    
    if filters.get('comorbidity_range'):
        min_comorb, max_comorb = filters['comorbidity_range']
        filtered_df = filtered_df[
            (filtered_df['comorbidity_count'] >= min_comorb) & 
            (filtered_df['comorbidity_count'] <= max_comorb)
        ]
    
    return filtered_df

def get_age_groups(df):
    """Create age groups"""
    df = df.copy()
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 30, 50, 70, 100],
        labels=['<30', '30-50', '50-70', '70+']
    )
    return df
