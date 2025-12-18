# Quick Start Guide

## Installation

1. Install required packages:
```bash
pip install streamlit plotly pandas openpyxl
```

Or install all requirements:
```bash
pip install -r ../requirements.txt
```

## Running the Dashboard

From the project root (`C:\Users\aliaw\OneDrive\Desktop\OR`):

```bash
streamlit run dashboard/app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## Dashboard Navigation

- Use the sidebar to apply global filters
- Navigate between pages using the sidebar menu
- All charts are interactive - hover, zoom, and pan as needed

## Troubleshooting

If you encounter import errors:
- Make sure you're running from the correct directory
- Verify that `data_cleaned.xlsx` exists in the parent directory
- Check that all dependencies are installed

If charts don't display:
- Check the browser console for errors
- Verify the data file is accessible
- Ensure Plotly is properly installed
