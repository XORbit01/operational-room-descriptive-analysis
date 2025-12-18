"""
Streamlit app entry point for Streamlit Cloud deployment
This file is used when deploying to Streamlit Community Cloud
"""

import sys
from pathlib import Path

# Add dashboard directory to path
dashboard_path = Path(__file__).parent / 'dashboard'
sys.path.insert(0, str(dashboard_path))

# Change to dashboard directory for imports
import os
os.chdir(dashboard_path)

# Import and run the main app
from app import *
