"""
Main Streamlit Dashboard Application
"""

import streamlit as st
from dashboard.utils.data_loader import load_data, apply_filters, get_age_groups
from dashboard.utils.styles import apply_custom_css

# Page configuration
st.set_page_config(
    page_title="OR Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("Global Filters")

# Gender filter
gender_options = df['Gender'].dropna().unique().tolist()
selected_gender = st.sidebar.multiselect(
    "Gender",
    options=gender_options,
    default=gender_options
)

# Age range filter
age_min = int(df['Age'].min())
age_max = int(df['Age'].max())
age_range = st.sidebar.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

# Governorate filter
governorate_options = df['Governorate'].dropna().unique().tolist()
selected_governorate = st.sidebar.multiselect(
    "Governorate",
    options=governorate_options,
    default=governorate_options
)

# BMI Category filter
bmi_options = df['BMI_Category'].dropna().unique().tolist()
selected_bmi = st.sidebar.multiselect(
    "BMI Category",
    options=bmi_options,
    default=bmi_options
)

# Emergency Status filter
emergency_options = df['Emergency Status of surgery'].dropna().unique().tolist()
selected_emergency = st.sidebar.multiselect(
    "Emergency Status",
    options=emergency_options,
    default=emergency_options
)

# Comorbidity count range
comorb_min = int(df['comorbidity_count'].min())
comorb_max = int(df['comorbidity_count'].max())
comorb_range = st.sidebar.slider(
    "Comorbidity Count",
    min_value=comorb_min,
    max_value=comorb_max,
    value=(comorb_min, comorb_max)
)

# Apply filters
filters = {
    'gender': selected_gender,
    'age_range': age_range,
    'governorate': selected_governorate,
    'bmi_category': selected_bmi,
    'emergency_status': selected_emergency,
    'comorbidity_range': comorb_range
}

filtered_df = apply_filters(df, filters)

# Store filtered data in session state
st.session_state['filtered_df'] = filtered_df
st.session_state['original_df'] = df

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(f"**Filtered Patients:** {len(filtered_df)} / {len(df)}")
