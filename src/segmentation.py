"""
segmentation.py
===============
Customer Segmentation for Churn Analytics.

Uses clustering on behavioral features to identify
distinct customer segments.

Author: Senior Data Scientist
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import config
try:
    from config import OUTPUT_DIR
except ImportError:
    from src.config import OUTPUT_DIR

PLOTS_DIR = OUTPUT_DIR / 'plots'


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features for segmentation
SEGMENTATION_FEATURES = [
    'total_sessions_30d',
    'avg_session_minutes_30d',
    'failed_payments_30d',
    'support_tickets_30d',
    'monthly_price'
]

# Segment labels
SEGMENT_LABELS = {
    0: 'High Value Engaged',
    1: 'At Risk',
    2: 'Low Engagement',
    3: 'Price Sensitive'
}


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def prepare_features_for_clustering(df: pd.DataFrame, 
                                     features: List[str] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Prepare features for clustering by scaling.
    
    Args:
        df: Input DataFrame
        features: Feature columns to use
    
    Returns:
        (scaled_features, scaler)
    """
    if features is None:
        features = SEGMENTATION_FEATURES
    
    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, available_features


def apply_pca(X_scaled: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA for visualization.
    
    Args:
        X_scaled: Scaled feature array
        n_components: Number of PCA components
    
    Returns:
        (pca_transformed, pca_model)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA with {n_components} components explains {explained_var:.1f}% of variance")
    
    return X_pca, pca


def find_optimal_k(X_scaled: np.ndarray, k_range: range = None) -> Tuple[int, List[float]]:
    """
    Find optimal k using elbow method.
    
    Args:
        X_scaled: Scaled feature array
        k_range: Range of k values to try
    
    Returns:
        (optimal_k, inertias)
    """
    if k_range is None:
        k_range = range(2, 10)
    
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection (find where decrease slows down)
    decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    
    # Find where decrease becomes less than 50% of previous
    optimal_k = 4  # Default
    for i in range(1, len(decreases)):
        if decreases[i] < decreases[i-1] * 0.5:
            optimal_k = list(k_range)[i]
            break
    
    return optimal_k, inertias


def perform_kmeans_clustering(X_scaled: np.ndarray, 
                               k: int = 4) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-Means clustering.
    
    Args:
        X_scaled: Scaled feature array
        k: Number of clusters
    
    Returns:
        (cluster_labels, kmeans_model)
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"K-Means clustering with k={k}")
    print(f"Cluster distribution:")
    for i in range(k):
        count = np.sum(labels == i)
        print(f"  Cluster {i}: {count:,} customers ({count/len(labels)*100:.1f}%)")
    
    return labels, kmeans


def segment_customers(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Full segmentation pipeline.
    
    Args:
        df: Input DataFrame with customer data
        k: Number of segments
    
    Returns:
        DataFrame with customer_id and segment_id
    """
    # Prepare features
    X_scaled, scaler, features = prepare_features_for_clustering(df)
    
    # Perform clustering
    labels, kmeans = perform_kmeans_clustering(X_scaled, k)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'customer_id': df['customer_id'] if 'customer_id' in df.columns else range(len(df)),
        'segment_id': labels
    })
    
    # Add segment names
    result['segment_name'] = result['segment_id'].map(
        lambda x: SEGMENT_LABELS.get(x, f'Segment_{x}')
    )
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_elbow_curve(inertias: List[float], k_range: range = None,
                     plots_dir: Path = None) -> Path:
    """
    Plot elbow curve for k selection.
    
    Args:
        inertias: List of inertia values
        k_range: Range of k values
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if k_range is None:
        k_range = range(2, 2 + len(inertias))
    
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(list(k_range), inertias, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    ax.grid(True, alpha=0.3)
    
    filepath = plots_dir / 'elbow_curve.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: elbow_curve.png")
    return filepath


def plot_pca_clusters(X_pca: np.ndarray, labels: np.ndarray,
                      plots_dir: Path = None) -> Path:
    """
    Plot PCA scatter plot with cluster labels.
    
    Args:
        X_pca: PCA-transformed data (n_samples, 2)
        labels: Cluster labels
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color palette
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f']
    
    # Plot each cluster
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        segment_name = SEGMENT_LABELS.get(label, f'Segment_{label}')
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[i % len(colors)], label=segment_name,
                   alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Customer Segments (PCA Visualization)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    filepath = plots_dir / 'pca_clusters.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: pca_clusters.png")
    return filepath


def plot_segment_profiles(df: pd.DataFrame, segments: pd.DataFrame,
                          features: List[str] = None,
                          plots_dir: Path = None) -> Path:
    """
    Plot radar chart or bar chart of segment profiles.
    
    Args:
        df: Original DataFrame with features
        segments: DataFrame with segment assignments
        features: Features to include in profile
        plots_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    if features is None:
        features = SEGMENTATION_FEATURES
    
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge segments with features
    merged = df.merge(segments, on='customer_id', how='left')
    
    # Compute mean of each feature per segment
    profiles = merged.groupby('segment_id')[features].mean()
    
    # Normalize for visualization
    profiles_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(features))
    width = 0.2
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, segment_id in enumerate(profiles_norm.index):
        segment_name = SEGMENT_LABELS.get(segment_id, f'Segment_{segment_id}')
        ax.bar(x + i * width, profiles_norm.loc[segment_id].values,
               width, label=segment_name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Segment Profiles')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    filepath = plots_dir / 'segment_profiles.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  ✓ Saved: segment_profiles.png")
    return filepath


def save_segments(segments: pd.DataFrame, filepath: Path = None) -> Path:
    """Save segment assignments to CSV."""
    if filepath is None:
        filepath = OUTPUT_DIR / 'segments.csv'
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(filepath, index=False)
    print(f"✓ Saved segments: {filepath}")
    
    return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Segmentation module loaded successfully.")
    print(f"\nSegmentation features: {SEGMENTATION_FEATURES}")
    print(f"\nSegment labels: {SEGMENT_LABELS}")
