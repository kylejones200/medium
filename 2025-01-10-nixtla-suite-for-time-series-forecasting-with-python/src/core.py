"""Core functions for Nixtla StatsForecast time series forecasting."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

os.environ['NIXTLA_ID_AS_COL'] = '1'

def generate_synthetic_data(start_date: str = '2023-01-01', periods: int = 240,
                            freq: str = 'H', seed: int = 42) -> pd.DataFrame:
    """Generate synthetic ERCOT-style time series data."""
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    load = np.sin(np.linspace(0, 12 * np.pi, periods)) * 20 + 100 + np.random.normal(0, 5, periods)
    df = pd.DataFrame({'ds': date_range, 'y': load})
    df["unique_id"] = "series1"
    return df

def split_data(df: pd.DataFrame, hold_out_hours: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and hold-out sets."""
    train = df.iloc[:-hold_out_hours]
    hold_out = df.iloc[-hold_out_hours:]
    return train, hold_out

def fit_statsforecast(train: pd.DataFrame, season_length: int = 24,
                     freq: str = 'h', n_jobs: int = -1) -> StatsForecast:
    """Fit StatsForecast model."""
    sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], 
                      freq=freq, n_jobs=n_jobs)
    sf.fit(train)
    return sf

def generate_forecast(sf: StatsForecast, horizon: int) -> pd.DataFrame:
    """Generate forecast for specified horizon."""
    forecasts = sf.predict(h=horizon)
    return forecasts

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate forecast evaluation metrics."""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

def plot_forecast(df: pd.DataFrame, hold_out: pd.DataFrame, forecasts: pd.DataFrame,
                 output_path: Path, model_name: str = "AutoARIMA", 
                 metrics: Dict[str, float] = None):
 """Plot forecast """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['ds'], df['y'], label='Historical Data', color="#4A90A4", linewidth=1.2)
    ax.plot(hold_out['ds'], hold_out['y'], label='Hold-Out Data', color="#8B6F9E", linewidth=1.2)
    ax.plot(forecasts['ds'], forecasts[model_name], label='Forecast', color="#D4A574", linewidth=1.2)
    
    title_text = f"Time Series Forecast with {model_name}"
    if metrics:
        title_text += f": RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}"
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

