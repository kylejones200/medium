"""Core functions for digital integration oil production monitoring systems."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze_production_monitoring(df: pd.DataFrame, timestamp_col: str,
                                 production_col: str) -> pd.DataFrame:
    """Analyze oil production monitoring data."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    return df

def calculate_production_kpis(df: pd.DataFrame, production_col: str) -> Dict:
    """Calculate production key performance indicators."""
    return {
        'total_production': df[production_col].sum(),
        'mean_production': df[production_col].mean(),
        'peak_production': df[production_col].max(),
        'efficiency': df[production_col].mean() / df[production_col].max() if df[production_col].max() > 0 else 0,
        'volatility': df[production_col].std() / df[production_col].mean() if df[production_col].mean() > 0 else 0
    }

def plot_production_monitoring(df: pd.DataFrame, production_col: str, title: str, output_path: Path):
 """Plot production monitoring """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df[production_col], color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Production")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

