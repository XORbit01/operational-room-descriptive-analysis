"""
Data loading utilities for the dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import os

# Try multiple possible paths for the data file
def get_data_path():
    """Get the path to the data file, trying multiple locations"""
    # Get the project root (where app.py is located)
    # Since app.py is at root, we can use current working directory or calculate from this file
    possible_paths = [
        # Path relative to this file (dashboard/utils/data_loader.py -> root/data/processed/)
        Path(__file__).parent.parent.parent / 'data' / 'processed' / 'data_cleaned.xlsx',
        # Path relative to current working directory (when running from root)
        Path.cwd() / 'data' / 'processed' / 'data_cleaned.xlsx',
        # Absolute path from current working directory
        Path('data') / 'processed' / 'data_cleaned.xlsx',
    ]
    
    # Try to find existing file
    for path in possible_paths:
        if path.exists():
            return path
    
    # If no path found, return the most likely one (relative to this file)
    # This will be used for error messages
    return Path(__file__).parent.parent.parent / 'data' / 'processed' / 'data_cleaned.xlsx'

DATA_PATH = get_data_path()

@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    if not DATA_PATH.exists():
        st.error(f"""
        **Data file not found!**
        
        The dashboard is looking for the data file at:
        `{DATA_PATH}`
        
        **For local development:**
        1. Run the data cleaning script: `python scripts/data_cleaning.py`
        2. This will generate `data/processed/data_cleaned.xlsx`
        
        **For Streamlit Cloud deployment:**
        1. Ensure `data/processed/data_cleaned.xlsx` is committed to your repository
        2. Or add the data file to your Streamlit Cloud app's file system
        3. Check that the file is not in `.gitignore`
        """)
        st.stop()
    
    try:
        df = pd.read_excel(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data file: {str(e)}")
        st.stop()

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
