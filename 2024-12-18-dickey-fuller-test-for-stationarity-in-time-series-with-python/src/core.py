"""Core functions for Dickey-Fuller stationarity testing."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_random_walk(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic random walk time series."""
    np.random.seed(seed)
    x = np.cumsum(np.random.normal(loc=0, scale=1, size=n_samples))
    return pd.DataFrame({'value': x})

def calculate_rolling_stats(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Calculate rolling mean and standard deviation."""
    return pd.DataFrame({
        'value': df['value'],
        'rolling_mean': df['value'].rolling(window=window).mean(),
        'rolling_std': df['value'].rolling(window=window).std()
    })

def test_stationarity(series: pd.Series) -> Dict[str, Any]:
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(series)
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }

def plot_time_series(df: pd.DataFrame, output_path: Path):
 """Plot time series """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df['value'], color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_rolling_stats(stats_df: pd.DataFrame, output_path: Path):
    """Plot time series with rolling statistics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(stats_df.index, stats_df['value'], label='Original', color="#4A90A4", linewidth=1.2)
    ax.plot(stats_df.index, stats_df['rolling_mean'], label='Rolling Mean', color="#D4A574", linewidth=1.2)
    ax.plot(stats_df.index, stats_df['rolling_std'], label='Rolling Std', color="#8B6F9E", linewidth=1.2)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

