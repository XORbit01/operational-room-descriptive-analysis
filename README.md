# Data Cleaning and Preprocessing Pipeline

This project implements a comprehensive data cleaning and preprocessing pipeline for a medical/surgical dataset, followed by correlation analysis.

---

<div align="center">
  <h3>📊 Dashboard Overview</h3>
  
  ![Dashboard Overview](overview/overview_1.png)
  
  <p><em>Interactive Streamlit Dashboard - Comprehensive Medical Data Analysis Interface</em></p>
</div>

---

## Dataset Overview

- **Original Size**: 522 rows × 139 columns
- **Final Cleaned Size**: 522 rows × 143 columns (after feature engineering)
- **Data Types**: 32 numerical, 105 categorical, 2 datetime columns

## Pipeline Structure

### 1. Data Cleaning (`data_cleaning.py`)

Implements 9 phases of data cleaning:

1. **Initial Data Loading and Assessment**
   - Loads data from `data.xlsx`
   - Creates backup (`data_backup.xlsx`)
   - Generates initial data quality report

2. **Column Name Standardization**
   - Fixes typos (e.g., "befor eDischarge" → "before Discharge")
   - Standardizes naming conventions

3. **Missing Value Handling (Feature-Specific Strategy)**
   - **Group A (Demographics)**: KNN imputation (k=5) for Weight, Height, BMI
   - **Group B (Medical History)**: Mode imputation for binary columns
   - **Group C (Lab Values)**: Subset analysis - complete cases only
   - **Group D (Post-Discharge)**: Missing indicators (truly unknown)
   - **Group E (Surgical Details)**: Context-specific handling
   - **Group F**: Drop high missing columns (>80%)

4. **Data Type Corrections**
   - Fixes mixed-type columns (BloodGroup)
   - Converts numeric to categorical where appropriate
   - Standardizes categorical values

5. **Outlier Detection and Treatment**
   - IQR method for outlier detection
   - Medical range validation and capping
   - Creates outlier flag columns

6. **Feature Engineering**
   - Comorbidity count
   - Medication count
   - Complication count
   - Lab value deltas (pre vs post-surgery)
   - Time-based features (days admission to surgery)

7. **Data Validation**
   - Age, Weight, Height range validation
   - BMI consistency checks
   - Date logic validation

8. **Prepare Analysis-Ready Datasets**
   - Numerical dataset
   - Categorical dataset
   - Mixed dataset

9. **Export Cleaned Data**
   - Main cleaned dataset
   - Lab subset datasets
   - Follow-up subset

### 2. Data Preprocessing (`data_preprocessing.py`)

- **Binary Encoding**: Yes/no variables → 0/1
- **One-Hot Encoding**: Nominal categories with <10 levels
- **Multicollinearity Handling**: Removes highly correlated pairs (>0.95)
- **Correlation-Ready Dataset**: Prepares numerical dataset for correlation analysis

### 3. Correlation Analysis (`correlation_analysis.py`)

- Computes Pearson and Spearman correlation matrices
- Identifies strong correlations (|r| >= 0.5)
- Generates correlation heatmaps
- Creates insights report

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed file organization.

### Key Files Generated

**Cleaned Datasets** (`data/processed/`):
- `data_cleaned.xlsx` - Main cleaned dataset
- `data_encoded.xlsx` - Encoded dataset for analysis
- `data_correlation_ready.xlsx` - Final dataset for correlation analysis
- `data_numerical.xlsx` - Numerical variables only
- `data_categorical.xlsx` - Categorical variables only

**Subset Datasets** (`data/processed/`):
- `data_pre_labs_subset.xlsx` - Patients with complete pre-surgery labs (276 patients)
- `data_post_labs_subset.xlsx` - Patients with Day 1 post-op labs (56 patients)
- `data_discharge_labs_subset.xlsx` - Patients with complete discharge labs (32 patients)
- `data_followup_subset.xlsx` - Patients with post-discharge follow-up (310 patients)

**Reports and Logs** (`reports/`):
- `data_quality_report_initial.txt` - Initial data quality assessment
- `data_quality_report_final.txt` - Final data quality assessment
- `imputation_log.txt` - Log of all imputation operations
- `encoding_log.txt` - Log of encoding operations
- `validation_issues.txt` - Data validation issues found

**Correlation Analysis Outputs** (`reports/`):
- `correlation_matrix_pearson.xlsx` - Full Pearson correlation matrix
- `correlation_matrix_spearman.xlsx` - Full Spearman correlation matrix
- `strong_correlations_pearson.xlsx` - Strong Pearson correlations (|r| >= 0.5)
- `strong_correlations_spearman.xlsx` - Strong Spearman correlations (|r| >= 0.5)
- `correlation_heatmap_pearson.png` - Pearson correlation heatmap visualization
- `correlation_heatmap_spearman.png` - Spearman correlation heatmap visualization
- `correlation_insights.txt` - Key insights from correlation analysis

## Usage

### Run Complete Pipeline

```bash
# 1. Data Cleaning
python scripts/data_cleaning.py

# 2. Data Preprocessing
python scripts/data_preprocessing.py

# 3. Correlation Analysis
python scripts/correlation_analysis.py

# 4. Dashboard
streamlit run dashboard/app.py
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Key Findings

### Strong Correlations Identified

**Top Pearson Correlations (|r| >= 0.5):**
- Complication Post Surgery ↔ complication_count: r = 0.93
- Blood Transfusion During Surgery ↔ Number of transfused PC: r = 0.90
- DiabetesMellitus ↔ Antidiabetic: r = 0.88
- Weight ↔ BMI: r = 0.82
- Pre-Creatinine ↔ Creatinine_D1: r = 0.76

**Top Spearman Correlations (|r| >= 0.5):**
- Blood Transfusion During Surgery ↔ Number of transfused PC: r = 1.00
- Complication Post Surgery ↔ complication_count: r = 1.00
- Current Medication ↔ medication_count: r = 0.96
- Weight ↔ BMI: r = 0.86
- comorbidity_count ↔ medication_count: r = 0.84

### Data Quality Improvements

- **Missing Values**: Handled using feature-specific strategies
- **Outliers**: Detected and capped at medical ranges
- **Data Types**: Standardized and corrected
- **Multicollinearity**: Removed 2 highly correlated pairs

## Configuration

All parameters and thresholds are defined in `config.py`:
- Missing value thresholds
- KNN imputation parameters
- Medical ranges for validation
- Columns to drop
- Feature groups

## Notes

- Lab values are **never imputed** - subset analysis is used instead
- Post-discharge outcomes preserve uncertainty (missing indicators)
- BMI is recalculated from Weight/Height to ensure consistency
- All transformations are logged for reproducibility
