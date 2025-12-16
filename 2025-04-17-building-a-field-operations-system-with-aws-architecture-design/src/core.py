"""Core functions for building field operations system with AWS architecture."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_field_operations_data(n_points: int, n_assets: int = 5, seed: int = 42) -> pd.DataFrame:
    """Simulate field operations data."""
    np.random.seed(seed)
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq='1H')
    
    data = {'timestamp': timestamps}
    for i in range(n_assets):
        base_value = 100 + i * 20
        noise = np.random.normal(0, 10, n_points)
        data[f'asset_{i+1}'] = base_value + noise
    
    return pd.DataFrame(data)

def analyze_field_operations(df: pd.DataFrame, asset_cols: list) -> Dict:
    """Analyze field operations data."""
    return {
        'n_samples': len(df),
        'n_assets': len(asset_cols),
        'mean_values': df[asset_cols].mean().to_dict(),
        'std_values': df[asset_cols].std().to_dict()
    }

def plot_field_operations(df: pd.DataFrame, asset_cols: list, title: str, output_path: Path):
 """Plot field operations data """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, col in enumerate(asset_cols[:5]):
        ax.plot(df['timestamp'], df[col], label=col, linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best', ncol=2)
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

