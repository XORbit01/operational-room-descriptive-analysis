"""
Correlation analysis and insights generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("=" * 80)
print("CORRELATION ANALYSIS AND INSIGHTS GENERATION")
print("=" * 80)

# Set up paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = Path(__file__).parent.parent / 'reports'

# Load correlation-ready dataset
print("\nLoading correlation-ready dataset...")
df = pd.read_excel(PROCESSED_DIR / 'data_correlation_ready.xlsx')
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================================================
# Correlation Matrices
# ============================================================================

print("\n[CORRELATION] Computing correlation matrices...")

# Pearson correlation (for linear relationships)
print("  Computing Pearson correlation...")
corr_pearson = df.corr(method='pearson')
print(f"    Pearson correlation matrix: {corr_pearson.shape}")

# Spearman correlation (for monotonic relationships)
print("  Computing Spearman correlation...")
corr_spearman = df.corr(method='spearman')
print(f"    Spearman correlation matrix: {corr_spearman.shape}")

# ============================================================================
# Identify Strong Correlations
# ============================================================================

print("\n[ANALYSIS] Identifying strong correlations...")

def find_strong_correlations(corr_matrix, threshold=0.5, method_name=''):
    """Find pairs of variables with strong correlations"""
    # Get upper triangle
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find strong correlations
    strong_corr = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            if pd.notna(upper_triangle.loc[idx, col]):
                corr_val = upper_triangle.loc[idx, col]
                if abs(corr_val) >= threshold:
                    strong_corr.append({
                        'Variable 1': idx,
                        'Variable 2': col,
                        'Correlation': corr_val,
                        'Method': method_name
                    })
    
    return pd.DataFrame(strong_corr).sort_values('Correlation', key=abs, ascending=False)

# Find strong Pearson correlations
strong_pearson = find_strong_correlations(corr_pearson, threshold=0.5, method_name='Pearson')
print(f"\n  Strong Pearson correlations (|r| >= 0.5): {len(strong_pearson)} pairs")

if len(strong_pearson) > 0:
    print("\n  Top 20 strongest Pearson correlations:")
    print(strong_pearson.head(20).to_string(index=False))

# Find strong Spearman correlations
strong_spearman = find_strong_correlations(corr_spearman, threshold=0.5, method_name='Spearman')
print(f"\n  Strong Spearman correlations (|r| >= 0.5): {len(strong_spearman)} pairs")

if len(strong_spearman) > 0:
    print("\n  Top 20 strongest Spearman correlations:")
    print(strong_spearman.head(20).to_string(index=False))

# ============================================================================
# Visualizations
# ============================================================================

print("\n[VISUALIZATION] Creating correlation visualizations...")

# Heatmap of Pearson correlation
print("  Creating Pearson correlation heatmap...")
plt.figure(figsize=(20, 16))
# Select top 30 most variable columns for readability
if len(corr_pearson.columns) > 30:
    # Select columns with highest variance
    variances = df.var().sort_values(ascending=False)
    top_cols = variances.head(30).index
    corr_subset = corr_pearson.loc[top_cols, top_cols]
else:
    corr_subset = corr_pearson

mask = np.triu(np.ones_like(corr_subset, dtype=bool))
sns.heatmap(corr_subset, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Pearson Correlation Matrix (Top 30 Variables)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'correlation_heatmap_pearson.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: reports/correlation_heatmap_pearson.png")

# Heatmap of Spearman correlation
print("  Creating Spearman correlation heatmap...")
plt.figure(figsize=(20, 16))
if len(corr_spearman.columns) > 30:
    corr_subset = corr_spearman.loc[top_cols, top_cols]
else:
    corr_subset = corr_spearman

mask = np.triu(np.ones_like(corr_subset, dtype=bool))
sns.heatmap(corr_subset, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Spearman Correlation Matrix (Top 30 Variables)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'correlation_heatmap_spearman.png', dpi=300, bbox_inches='tight')
plt.close()
print("    Saved: reports/correlation_heatmap_spearman.png")

# ============================================================================
# Key Insights
# ============================================================================

print("\n[INSIGHTS] Generating key insights...")

insights = []

# 1. Demographic correlations
demo_cols = [col for col in df.columns if col in ['Age', 'Weight', 'Height', 'BMI']]
if demo_cols:
    demo_corr = corr_pearson.loc[demo_cols, demo_cols]
    insights.append("DEMOGRAPHIC CORRELATIONS:")
    insights.append(str(demo_corr))
    insights.append("")

# 2. Medical history correlations
med_cols = [col for col in df.columns if any(term in col.lower() for term in 
            ['hypertension', 'diabetes', 'cad', 'copd', 'ckd', 'comorbidity'])]
if med_cols:
    med_corr = corr_pearson.loc[med_cols[:10], med_cols[:10]]  # Limit to 10x10
    insights.append("MEDICAL HISTORY CORRELATIONS:")
    insights.append(str(med_corr))
    insights.append("")

# 3. Outcome correlations
outcome_cols = [col for col in df.columns if any(term in col.lower() for term in 
                ['complication', 'death', 'readmission', 'duration'])]
if outcome_cols:
    outcome_corr = corr_pearson.loc[outcome_cols[:10], outcome_cols[:10]]
    insights.append("OUTCOME CORRELATIONS:")
    insights.append(str(outcome_corr))
    insights.append("")

# Save insights
with open(REPORTS_DIR / 'correlation_insights.txt', 'w') as f:
    f.write("CORRELATION ANALYSIS INSIGHTS\n")
    f.write("=" * 80 + "\n\n")
    f.write("\n".join(insights))
    f.write("\n\nTOP STRONG CORRELATIONS (Pearson |r| >= 0.5):\n")
    f.write("=" * 80 + "\n")
    f.write(strong_pearson.to_string(index=False))
    f.write("\n\nTOP STRONG CORRELATIONS (Spearman |r| >= 0.5):\n")
    f.write("=" * 80 + "\n")
    f.write(strong_spearman.to_string(index=False))

print("    Saved: reports/correlation_insights.txt")

# Save correlation matrices
print("\n[SAVING] Saving correlation matrices...")
corr_pearson.to_excel(REPORTS_DIR / 'correlation_matrix_pearson.xlsx')
print("  Saved: reports/correlation_matrix_pearson.xlsx")

corr_spearman.to_excel(REPORTS_DIR / 'correlation_matrix_spearman.xlsx')
print("  Saved: reports/correlation_matrix_spearman.xlsx")

strong_pearson.to_excel(REPORTS_DIR / 'strong_correlations_pearson.xlsx', index=False)
print("  Saved: reports/strong_correlations_pearson.xlsx")

strong_spearman.to_excel(REPORTS_DIR / 'strong_correlations_spearman.xlsx', index=False)
print("  Saved: reports/strong_correlations_spearman.xlsx")

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - reports/correlation_heatmap_pearson.png")
print("  - reports/correlation_heatmap_spearman.png")
print("  - reports/correlation_matrix_pearson.xlsx")
print("  - reports/correlation_matrix_spearman.xlsx")
print("  - reports/strong_correlations_pearson.xlsx")
print("  - reports/strong_correlations_spearman.xlsx")
print("  - reports/correlation_insights.txt")
