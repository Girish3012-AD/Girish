"""
eda_utils.py
============
Utility functions for Exploratory Data Analysis.

Contains functions for visualization and statistical analysis.

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Import project config
try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR

PLOTS_DIR = OUTPUT_DIR / 'plots'


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def setup_plot_style():
    """Setup matplotlib style for consistent plots."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def save_plot(fig: plt.Figure, filename: str, plots_dir: Path = None) -> Path:
    """Save plot to file."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = plots_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")
    return filepath


def plot_churn_distribution(df: pd.DataFrame, target_col: str = 'churn',
                            plots_dir: Path = None) -> Path:
    """
    Plot churn distribution bar chart.
    
    Args:
        df: DataFrame with churn column
        target_col: Name of target column
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = df[target_col].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']  # Green for 0, Red for 1
    labels = ['Not Churned (0)', 'Churned (1)']
    
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Churn Status')
    ax.set_ylabel('Count')
    ax.set_title('Customer Churn Distribution')
    ax.set_ylim(0, max(counts.values) * 1.15)
    
    return save_plot(fig, 'churn_distribution.png', plots_dir)


def plot_correlation_heatmap(df: pd.DataFrame, plots_dir: Path = None) -> Path:
    """
    Plot correlation heatmap for numeric features.
    
    Args:
        df: DataFrame with numeric columns
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    setup_plot_style()
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap using imshow
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    
    # Set ticks
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)
    
    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            value = corr_matrix.iloc[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=8)
    
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    return save_plot(fig, 'correlation_heatmap.png', plots_dir)


def plot_boxplots_vs_churn(df: pd.DataFrame, numeric_cols: List[str],
                           target_col: str = 'churn', plots_dir: Path = None) -> Path:
    """
    Plot boxplots of numeric features vs churn.
    
    Args:
        df: DataFrame
        numeric_cols: List of numeric column names
        target_col: Target column name
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    setup_plot_style()
    
    n_cols = min(len(numeric_cols), 6)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    colors = ['#2ecc71', '#e74c3c']
    
    for idx, col in enumerate(numeric_cols):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # Prepare data for boxplot
        data_0 = df[df[target_col] == 0][col].dropna()
        data_1 = df[df[target_col] == 1][col].dropna()
        
        bp = ax.boxplot([data_0, data_1], labels=['Not Churned', 'Churned'],
                        patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(col, fontsize=10)
        ax.set_ylabel('Value')
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Feature Distributions by Churn Status', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return save_plot(fig, 'boxplots_vs_churn.png', plots_dir)


def plot_histograms(df: pd.DataFrame, columns: List[str],
                    plots_dir: Path = None) -> Path:
    """
    Plot histograms for specified columns.
    
    Args:
        df: DataFrame
        columns: List of column names
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    setup_plot_style()
    
    n_cols = min(len(columns), 3)
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        data = df[col].dropna()
        ax.hist(data, bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {col}')
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    return save_plot(fig, 'feature_histograms.png', plots_dir)


def plot_churn_rate_by_category(df: pd.DataFrame, cat_cols: List[str],
                                 target_col: str = 'churn',
                                 plots_dir: Path = None) -> Path:
    """
    Plot churn rate by categorical features.
    
    Args:
        df: DataFrame
        cat_cols: List of categorical column names
        target_col: Target column name
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    setup_plot_style()
    
    n_cols = min(len(cat_cols), 3)
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for idx, col in enumerate(cat_cols):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        churn_rate = df.groupby(col)[target_col].mean() * 100
        churn_rate = churn_rate.sort_values(ascending=False)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(churn_rate)))
        bars = ax.bar(range(len(churn_rate)), churn_rate.values, color=colors, edgecolor='black')
        
        ax.set_xticks(range(len(churn_rate)))
        ax.set_xticklabels(churn_rate.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Churn Rate (%)')
        ax.set_title(f'Churn Rate by {col}')
        
        # Add value labels
        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Hide empty subplots
    for idx in range(len(cat_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    return save_plot(fig, 'churn_rate_by_category.png', plots_dir)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic statistics for numeric columns.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with stats
    """
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats['missing'] = df[numeric_df.columns].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df) * 100).round(2)
    return stats


def compute_unique_values(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, List]:
    """
    Compute unique values for categorical columns.
    
    Args:
        df: Input DataFrame
        cat_cols: List of categorical columns
    
    Returns:
        Dictionary mapping column to unique values
    """
    return {col: df[col].unique().tolist() for col in cat_cols if col in df.columns}


if __name__ == "__main__":
    print("EDA Utilities module loaded successfully.")
