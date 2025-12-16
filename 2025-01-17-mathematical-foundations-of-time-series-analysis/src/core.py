"""Core functions for mathematical foundations of time series analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_statistical_properties(series: pd.Series) -> Dict:
    """Calculate fundamental statistical properties."""
    return {
        'mean': series.mean(),
        'variance': series.var(),
        'std': series.std(),
        'skewness': series.skew(),
        'kurtosis': series.kurtosis(),
        'autocorr_lag1': series.autocorr(lag=1)
    }

def calculate_autocorrelation(series: pd.Series, max_lag: int = 10) -> pd.Series:
    """Calculate autocorrelation function."""
    autocorrs = []
    for lag in range(1, max_lag + 1):
        autocorrs.append(series.autocorr(lag=lag))
    return pd.Series(autocorrs, index=range(1, max_lag + 1))

def plot_mathematical_properties(series: pd.Series, acf: pd.Series, title: str, output_path: Path):
 """Plot mathematical properties """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    
    ax1.plot(series.index, series.values, color="#4A90A4", linewidth=1.2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    
    ax2.bar(acf.index, acf.values, color="#D4A574", alpha=0.7, edgecolor='none', width=0.6)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("ACF")
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

