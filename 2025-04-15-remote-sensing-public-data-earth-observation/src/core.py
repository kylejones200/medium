"""Core functions for remote sensing and public data earth observation."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_satellite_data(n_points: int, n_bands: int = 3, seed: int = 42) -> pd.DataFrame:
    """Simulate satellite remote sensing data."""
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
    
    data = {'date': dates}
    band_names = ['red', 'green', 'blue', 'nir', 'swir']
    for i, band in enumerate(band_names[:n_bands]):
        base = 100 + i * 50
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
        noise = np.random.normal(0, 5, n_points)
        data[band] = base + seasonal + noise
    
    return pd.DataFrame(data)

def analyze_earth_observation(df: pd.DataFrame, band_cols: list) -> Dict:
    """Analyze earth observation data."""
    return {
        'n_observations': len(df),
        'n_bands': len(band_cols),
        'mean_values': df[band_cols].mean().to_dict(),
        'ndvi': None
    }

def calculate_ndvi(df: pd.DataFrame, red_col: str, nir_col: str) -> pd.Series:
    """Calculate Normalized Difference Vegetation Index."""
    if red_col in df.columns and nir_col in df.columns:
        ndvi = (df[nir_col] - df[red_col]) / (df[nir_col] + df[red_col] + 1e-10)
        return ndvi
    return pd.Series()

def plot_earth_observation(df: pd.DataFrame, band_cols: list, title: str, output_path: Path):
 """Plot earth observation data """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for band in band_cols[:3]:
        ax.plot(df['date'], df[band], label=band.upper(), linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Reflectance")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

