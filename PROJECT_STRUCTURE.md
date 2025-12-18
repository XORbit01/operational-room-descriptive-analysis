# Project Structure

```
OR/
├── config/                 # Configuration files
│   └── config.py          # Data cleaning and preprocessing parameters
│
├── data/                   # Data files
│   ├── raw/               # Original/raw data files
│   │   ├── data.xlsx      # Original dataset
│   │   └── data_backup.xlsx
│   └── processed/         # Processed/cleaned data files
│       ├── data_cleaned.xlsx
│       ├── data_encoded.xlsx
│       ├── data_correlation_ready.xlsx
│       ├── data_numerical.xlsx
│       ├── data_categorical.xlsx
│       └── *_subset.xlsx  # Lab subset datasets
│
├── scripts/                # Analysis scripts
│   ├── data_cleaning.py   # Main data cleaning pipeline
│   ├── data_preprocessing.py  # Data preprocessing for analysis
│   ├── correlation_analysis.py  # Correlation analysis
│   └── data_quality_report.py  # Quality report generator
│
├── reports/                # Generated reports and outputs
│   ├── *.txt              # Log files and reports
│   ├── *.xlsx             # Correlation matrices
│   └── *.png              # Visualization images
│
├── dashboard/              # Streamlit dashboard
│   ├── app.py             # Main dashboard application
│   ├── pages/             # Dashboard pages
│   │   ├── 1_Overview.py
│   │   ├── 2_Demographics.py
│   │   ├── 3_Medical_History.py
│   │   ├── 4_Pre_Surgery.py
│   │   ├── 5_Surgery_Details.py
│   │   ├── 6_Post_Surgery.py
│   │   └── 7_Discharge_Followup.py
│   └── utils/             # Dashboard utilities
│       ├── data_loader.py
│       ├── charts.py
│       └── styles.py
│
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Running Scripts

All scripts should be run from the project root directory:

```bash
# Data cleaning
python scripts/data_cleaning.py

# Data preprocessing
python scripts/data_preprocessing.py

# Correlation analysis
python scripts/correlation_analysis.py

# Dashboard
streamlit run dashboard/app.py
```

## Data Flow

1. **Raw Data** → `data/raw/data.xlsx`
2. **Cleaning** → `scripts/data_cleaning.py` → `data/processed/data_cleaned.xlsx`
3. **Preprocessing** → `scripts/data_preprocessing.py` → `data/processed/data_correlation_ready.xlsx`
4. **Analysis** → `scripts/correlation_analysis.py` → `reports/*`
5. **Visualization** → `dashboard/app.py` → Reads from `data/processed/`
