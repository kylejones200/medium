"""Core functions for PCA and Mahalanobis distance in predictive maintenance."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_cmapss_data(data_path: Path, sep: str = r'\s+') -> pd.DataFrame:
    """Load CMAPSS dataset."""
    column_names = ['unit', 'time'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(data_path, sep=sep, header=None)
    df.dropna(axis=1, inplace=True)
    df.columns = column_names[:len(df.columns)]
    return df

def apply_pca(df: pd.DataFrame, sensor_columns: List[str], n_components: int = 3) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
    """Apply PCA to sensor data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[sensor_columns])
    pca = PCA(n_components=n_components)
    pca_factors = pca.fit_transform(X_scaled)
    
    for i in range(n_components):
        df[f'pca_{i+1}'] = pca_factors[:, i]
    
    return df, pca, scaler

def calculate_mahalanobis_distance(df: pd.DataFrame, pca_columns: List[str]) -> pd.Series:
    """Calculate Mahalanobis distance from origin."""
    pca_values = df[pca_columns].values
    distance = np.sqrt(np.sum(pca_values**2, axis=1))
    return pd.Series(distance, index=df.index)

def plot_health_index(df: pd.DataFrame, unit_id: int, distance_col: str,
                     threshold: float, output_path: Path):
 """Plot health index for a specific unit """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    unit_data = df[df['unit'] == unit_id]
    ax.plot(unit_data['time'], unit_data[distance_col], 
           color="#4A90A4", linewidth=1.2, alpha=0.8)
    ax.axhline(threshold, color="#D4A574", linestyle='--', 
              linewidth=1.2, label='Threshold')
    
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Health Index')
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
