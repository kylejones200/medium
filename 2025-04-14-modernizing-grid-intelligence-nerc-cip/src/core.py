"""Core functions for modernizing grid intelligence with NERC CIP compliance."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze_grid_data(df: pd.DataFrame, timestamp_col: str, value_col: str) -> pd.DataFrame:
    """Analyze grid intelligence data."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    return df

def calculate_grid_metrics(df: pd.DataFrame, value_col: str) -> Dict:
    """Calculate grid performance metrics."""
    return {
        'mean': df[value_col].mean(),
        'std': df[value_col].std(),
        'min': df[value_col].min(),
        'max': df[value_col].max(),
        'volatility': df[value_col].std() / df[value_col].mean()
    }

def plot_grid_intelligence(df: pd.DataFrame, value_col: str, title: str, output_path: Path):
 """Plot grid intelligence data """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df[value_col], color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

