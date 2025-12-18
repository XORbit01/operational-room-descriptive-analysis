"""
Streamlit app entry point for Streamlit Cloud deployment
This file is used when deploying to Streamlit Community Cloud
"""

import sys
from pathlib import Path

# Add dashboard directory to path
sys.path.insert(0, str(Path(__file__).parent / 'dashboard'))

# Import and run the main app
from app import *
