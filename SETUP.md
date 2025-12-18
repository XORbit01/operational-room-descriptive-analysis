# Setup and Quick Start Guide

## Initial Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Data File Location**
   - Original data should be at: `data/raw/data.xlsx`
   - If not present, place your `data.xlsx` file there

## Running the Complete Pipeline

### Step 1: Data Cleaning
```bash
python scripts/data_cleaning.py
```
**Outputs:**
- `data/processed/data_cleaned.xlsx` - Main cleaned dataset
- `data/processed/*_subset.xlsx` - Lab subset datasets
- `reports/data_quality_report_*.txt` - Quality reports
- `reports/imputation_log.txt` - Imputation log
- `reports/validation_issues.txt` - Validation issues

### Step 2: Data Preprocessing
```bash
python scripts/data_preprocessing.py
```
**Outputs:**
- `data/processed/data_encoded.xlsx` - Encoded dataset
- `data/processed/data_correlation_ready.xlsx` - Correlation-ready dataset
- `reports/encoding_log.txt` - Encoding log

### Step 3: Correlation Analysis
```bash
python scripts/correlation_analysis.py
```
**Outputs:**
- `reports/correlation_matrix_pearson.xlsx` - Pearson correlation matrix
- `reports/correlation_matrix_spearman.xlsx` - Spearman correlation matrix
- `reports/strong_correlations_*.xlsx` - Strong correlations
- `reports/correlation_heatmap_*.png` - Correlation heatmaps
- `reports/correlation_insights.txt` - Insights report

### Step 4: Launch Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard will open in your browser at `http://localhost:8501`

## File Locations Reference

| File Type | Location |
|-----------|----------|
| Original Data | `data/raw/data.xlsx` |
| Cleaned Data | `data/processed/data_cleaned.xlsx` |
| Correlation Data | `data/processed/data_correlation_ready.xlsx` |
| Scripts | `scripts/` |
| Reports/Logs | `reports/` |
| Dashboard | `dashboard/` |
| Config | `config/config.py` |

## Troubleshooting

**Issue**: Script can't find data file
- **Solution**: Ensure `data.xlsx` is in `data/raw/` directory

**Issue**: Import errors
- **Solution**: Run `pip install -r requirements.txt` from project root

**Issue**: Dashboard can't load data
- **Solution**: Run data cleaning script first to generate `data/processed/data_cleaned.xlsx`

**Issue**: Path errors in scripts
- **Solution**: Always run scripts from project root directory
