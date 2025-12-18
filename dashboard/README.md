# OR Data Analysis Dashboard

Professional Streamlit dashboard for comprehensive descriptive analysis of medical/surgical dataset.

## Features

- **7 Analysis Pages**: Overview, Demographics, Medical History, Pre-Surgery, Surgery Details, Post-Surgery, and Discharge/Follow-up
- **Global Filters**: Filter data by Gender, Age, Governorate, BMI Category, Emergency Status, and Comorbidity Count
- **Interactive Visualizations**: All charts are interactive using Plotly
- **Comprehensive Analysis**: Covers all aspects from patient demographics to post-discharge outcomes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

From the project root directory:

```bash
streamlit run dashboard/app.py
```

Or from the dashboard directory:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Dashboard Structure

### Main App (`app.py`)
- Sets up page configuration
- Loads data
- Provides global sidebar filters
- Stores filtered data in session state

### Pages

1. **Overview** - Key metrics, summary charts, outcome overview
2. **Demographics** - Age, gender, BMI, geographic, smoking analysis
3. **Medical History** - Comorbidity prevalence, medication analysis
4. **Pre-Surgery** - Lab values, clinical indicators, ER admissions
5. **Surgery Details** - Procedure types, duration, anesthesia, blood transfusion
6. **Post-Surgery** - Complications, ICU transfers, lab value changes
7. **Discharge/Follow-up** - Hospitalization duration, outcomes, predictors

### Utilities

- `data_loader.py` - Data loading and filtering functions
- `charts.py` - Reusable chart functions
- `styles.py` - Custom CSS styling

## Data Requirements

The dashboard expects `data_cleaned.xlsx` in the parent directory (same level as dashboard folder).

## Notes

- All filters are applied globally across all pages
- Data is cached for performance
- Charts are interactive - hover for details, zoom, pan, etc.
- Professional styling with no emojis as requested
