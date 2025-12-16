"""Core functions for Bollinger Bands analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_synthetic_prices(start_date: str = "2024-04-01", end_date: str = "2024-10-31",
                              freq: str = "B", initial_price: float = 2.5,
                              volatility: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic price data."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    prices = initial_price + np.cumsum(np.random.normal(0, volatility, len(dates)))
    return pd.DataFrame({"adjClose": prices}, index=dates)

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20,
                              num_std: float = 2.0, target_col: str = 'adjClose') -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    df = df.copy()
    df[f'{window} MA'] = df[target_col].rolling(window).mean()
    std = df[target_col].rolling(window).std()
    df['Lower'] = df[f'{window} MA'] - num_std * std
    df['Upper'] = df[f'{window} MA'] + num_std * std
    return df.dropna()

def plot_bollinger_bands(df: pd.DataFrame, target_col: str = 'adjClose',
                         window: int = 20, output_path: Path = None):
 """Plot Bollinger Bands """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(df.index, df['Lower'].values, df['Upper'].values,
                    alpha=0.15, color="#8B6F9E", label="Bollinger Band")
    ax.plot(df.index, df[target_col].values, label=target_col,
           color="#4A90A4", linewidth=1.2)
    ax.plot(df.index, df[f'{window} MA'].values, label=f"{window} Day MA",
           color="#D4A574", linewidth=1.2, linestyle="--")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc='best')
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

