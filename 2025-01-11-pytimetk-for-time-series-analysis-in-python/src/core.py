"""Core functions for PyTimeTK time series analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def prepare_time_series_data(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """Prepare time series data for PyTimeTK."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    return df[value_col]

def analyze_time_series_features(series: pd.Series) -> Dict:
    """Analyze time series features."""
    return {
        'length': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing',
        'volatility': series.pct_change().std()
    }

def plot_pytimetk_analysis(series: pd.Series, title: str, output_path: Path):
 """Plot PyTimeTK analysis """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(series.index, series.values, color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
