"""
Generate comprehensive data quality assessment report
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_quality_report(df, output_file='data_quality_report.txt'):
    """
    Generate a comprehensive data quality report
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    output_file : str
        Path to save the report
    """
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA QUALITY ASSESSMENT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Missing values summary
    report_lines.append("-" * 80)
    report_lines.append("MISSING VALUES SUMMARY")
    report_lines.append("-" * 80)
    
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).sort_values(ascending=False)
    
    report_lines.append(f"\nTotal columns with missing values: {(missing_counts > 0).sum()}")
    report_lines.append(f"Total missing cells: {missing_counts.sum()}")
    report_lines.append(f"Overall missing percentage: {(missing_counts.sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
    
    report_lines.append("\nTop 25 columns by missing percentage:")
    report_lines.append("-" * 80)
    for col, pct in missing_pct[missing_pct > 0].head(25).items():
        count = missing_counts[col]
        report_lines.append(f"{col:50s} {count:4d} ({pct:6.2f}%)")
    
    # Data types summary
    report_lines.append("\n" + "-" * 80)
    report_lines.append("DATA TYPES SUMMARY")
    report_lines.append("-" * 80)
    
    dtype_counts = df.dtypes.value_counts()
    report_lines.append("\nData type distribution:")
    for dtype, count in dtype_counts.items():
        report_lines.append(f"  {str(dtype):20s}: {count:3d} columns")
    
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    report_lines.append(f"\nNumerical columns: {len(num_cols)}")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    report_lines.append(f"Categorical columns: {len(cat_cols)}")
    
    # Datetime columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    report_lines.append(f"Datetime columns: {len(date_cols)}")
    
    # Categorical columns details
    report_lines.append("\n" + "-" * 80)
    report_lines.append("CATEGORICAL COLUMNS DETAILS")
    report_lines.append("-" * 80)
    
    for col in cat_cols[:15]:  # First 15 categorical columns
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        report_lines.append(f"\n{col}:")
        report_lines.append(f"  Unique values: {unique_count}")
        report_lines.append(f"  Missing: {missing_count} ({missing_count/len(df)*100:.1f}%)")
        if unique_count <= 10:
            value_counts = df[col].value_counts().head(5)
            report_lines.append(f"  Top values:")
            for val, count in value_counts.items():
                report_lines.append(f"    {val}: {count}")
    
    # Numerical columns statistics
    report_lines.append("\n" + "-" * 80)
    report_lines.append("NUMERICAL COLUMNS STATISTICS")
    report_lines.append("-" * 80)
    
    if len(num_cols) > 0:
        report_lines.append("\nKey numerical columns summary:")
        key_num_cols = ['Age', 'Weight', 'Height', 'BMI', 'Duration of hospitalization (days)']
        for col in key_num_cols:
            if col in num_cols:
                report_lines.append(f"\n{col}:")
                stats = df[col].describe()
                missing = df[col].isnull().sum()
                report_lines.append(f"  Missing: {missing} ({missing/len(df)*100:.1f}%)")
                report_lines.append(f"  Mean: {stats['mean']:.2f}")
                report_lines.append(f"  Median: {stats['50%']:.2f}")
                report_lines.append(f"  Min: {stats['min']:.2f}")
                report_lines.append(f"  Max: {stats['max']:.2f}")
                report_lines.append(f"  Std: {stats['std']:.2f}")
    
    # Duplicate rows
    report_lines.append("\n" + "-" * 80)
    report_lines.append("DUPLICATE ROWS")
    report_lines.append("-" * 80)
    duplicate_count = df.duplicated().sum()
    report_lines.append(f"\nTotal duplicate rows: {duplicate_count}")
    
    # Write report to file
    report_text = "\n".join(report_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Data quality report saved to: {output_file}")
    return report_text
