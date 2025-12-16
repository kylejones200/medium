"""Core functions for seasonal and trend decomposition."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_synthetic_data(start_date: str = '2024-01-01', end_date: str = '2025-12-31',
                           freq: str = 'D', seed: int = 42) -> pd.DataFrame:
    """Generate synthetic time series with known trend, seasonality, and noise."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    t = np.linspace(0, len(dates)-1, len(dates))
    trend = 0.1 * t + 10
    seasonality = 5 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 1, len(dates))
    
    data = trend + seasonality + noise
    df = pd.DataFrame({'date': dates, 'value': data})
    df.set_index('date', inplace=True)
    return df

def decompose_additive(df: pd.DataFrame, period: int = 365):
    """Perform additive seasonal decomposition."""
    return seasonal_decompose(df['value'], period=period, model='additive')

def decompose_multiplicative(df: pd.DataFrame, period: int = 365):
    """Perform multiplicative seasonal decomposition."""
    return seasonal_decompose(df['value'], period=period, model='multiplicative')

def robust_decomposition(df: pd.DataFrame, period: int = 365) -> pd.DataFrame:
    """Perform robust decomposition using Savitzky-Golay filter."""
    trend = pd.Series(
        savgol_filter(df['value'], window_length=period // 2 + 1, polyorder=3),
        index=df.index
    )
    detrended = df['value'] - trend
    seasonal = pd.Series(0.0, index=df.index)
    for i in range(period):
        seasonal[i::period] = detrended[i::period].median()
    residuals = df['value'] - trend - seasonal
    return pd.DataFrame({
        'original': df['value'],
        'trend': trend,
        'seasonal': seasonal,
        'residual': residuals
    })

def analyze_components(decomposition_result) -> Dict[str, Any]:
    """Analyze decomposition components."""
    trend = decomposition_result.trend.dropna()
    trend_direction = 'increasing' if trend.iloc[-1] > trend.iloc[0] else 'decreasing'
    trend_strength = abs(trend.iloc[-1] - trend.iloc[0]) / len(trend)
    
    seasonal = decomposition_result.seasonal.dropna()
    seasonal_amplitude = seasonal.max() - seasonal.min()
    
    residuals = decomposition_result.resid.dropna()
    residual_variance = residuals.var()
    
    return {
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'seasonal_amplitude': seasonal_amplitude,
        'residual_variance': residual_variance
    }

def plot_decomposition_comparison(df: pd.DataFrame, additive, multiplicative, output_path: Path):
    """Plot comparison of additive and multiplicative decomposition."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    axes[0, 0].plot(df['value'], color="#4A90A4", linewidth=1.2)
    axes[0, 1].plot(additive.trend, color="#4A90A4", linewidth=1.2)
    axes[0, 2].plot(additive.seasonal, color="#D4A574", linewidth=1.2)
    axes[0, 3].plot(additive.resid, color="#8B6F9E", linewidth=1.2)
    
    axes[1, 0].plot(df['value'], color="#4A90A4", linewidth=1.2)
    axes[1, 1].plot(multiplicative.trend, color="#4A90A4", linewidth=1.2)
    axes[1, 2].plot(multiplicative.seasonal, color="#D4A574", linewidth=1.2)
    axes[1, 3].plot(multiplicative.resid, color="#8B6F9E", linewidth=1.2)
    
    plt.suptitle("Additive vs Multiplicative Decomposition Comparison", 
                 fontsize=12, y=0.98, color='0.2')
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

