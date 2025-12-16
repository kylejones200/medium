"""Core functions for time series manipulation with Pandas."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def manipulate_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Perform common time series manipulations."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    df['rolling_mean'] = df[value_col].rolling(window=7).mean()
    df['rolling_std'] = df[value_col].rolling(window=7).std()
    df['shifted'] = df[value_col].shift(1)
    df['pct_change'] = df[value_col].pct_change()
    
    return df

def resample_time_series(df: pd.Series, freq: str = 'W') -> pd.Series:
    """Resample time series to different frequency."""
    return df.resample(freq).mean()

def plot_time_series_manipulation(df: pd.DataFrame, value_col: str, title: str, output_path: Path):
 """Plot time series manipulations """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(df.index, df[value_col], label="Original", color="#4A90A4", linewidth=1.2)
    axes[0].plot(df.index, df['rolling_mean'], label="Rolling Mean", color="#D4A574", linewidth=1.2)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc='best')
    
    axes[1].plot(df.index, df['pct_change'], label="Percent Change", color="#8B6F9E", linewidth=1.2)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Percent Change")
    axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

