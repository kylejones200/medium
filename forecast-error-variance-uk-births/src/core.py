"""Core functions for forecast error variance analysis using UK births data."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_and_prepare_data(file_path: Path = None, url: str = None) -> pd.DataFrame:
    """Load and prepare UK births data."""
    if file_path and file_path.exists():
        df = pd.read_excel(file_path, sheet_name='Sheet1')
    elif url:
        import urllib.request
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            df = pd.read_excel(tmp.name, sheet_name='Sheet1')
    else:
        dates = pd.date_range('2000-01-01', periods=100, freq='Q')
        births = 200000 + 10000 * np.sin(np.arange(100) / 4) + np.random.normal(0, 5000, 100)
        df = pd.DataFrame({'Date': dates, 'Births': births})
        return df.set_index('Date')[['Births']]
    
    df['Year'] = df['Year'].ffill()
    quarter_to_month = {'Mar': 3, 'Jun': 6, 'Sep': 9, 'Dec': 12}
    df['Month'] = df['Quarter'].map(quarter_to_month)
    df['Date'] = pd.to_datetime(dict(year=df['Year'].astype(int), month=df['Month'], day=1))
    df = df.set_index('Date')
    return df[['Births']].copy()

def fit_forecast_births(series: pd.Series, seasonal_periods: int = 4) -> pd.Series:
    """Fit exponential smoothing model and return fitted values."""
    model = ExponentialSmoothing(
        series,
        seasonal='add',
        trend='add',
        seasonal_periods=seasonal_periods,
        initialization_method="estimated"
    ).fit()
    return model.fittedvalues

def calculate_error_metrics(errors: pd.Series, alpha: float = 0.2) -> Dict:
    """Calculate forecast error metrics."""
    mad = errors.abs().mean()
    sigma_approx = mad * np.sqrt(np.pi / 2)
    sample_var = errors.var(ddof=1)
    
    return {
        'mean_error': errors.mean(),
        'mad': mad,
        'sigma_approx': sigma_approx,
        'sample_variance': sample_var
    }

def calculate_multi_step_variance(c1: float, c_tau: float, mad: float, alpha: float = 0.2) -> float:
    """Calculate multi-step forecast error variance."""
    sigma_e_approx = mad * np.sqrt(np.pi / 2)
    var_one_step = (sigma_e_approx**2) * (2 - alpha) / 2
    return c_tau * var_one_step

def plot_forecast_analysis(actual: pd.Series, fitted: pd.Series, errors: pd.Series,
                          title: str, output_path: Path):
 """Plot forecast analysis """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(actual.index, actual.values, label="Actual", color="#4A90A4", linewidth=1.2)
    axes[0].plot(fitted.index, fitted.values, label="Fitted", color="#D4A574", 
                linewidth=1.2, linestyle='--')
    axes[0].set_ylabel("Births")
    axes[0].legend(loc='best')
    
    axes[1].plot(errors.index, errors.values, color="#8B6F9E", linewidth=1.2)
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    axes[1].set_ylabel("Error")
    
    smoothed = errors.ewm(alpha=0.2, adjust=False).mean()
    axes[2].plot(errors.index, errors.rolling(20).mean(), label="Moving Average (20)",
                 color="#4A90A4", linewidth=1.2)
    axes[2].plot(errors.index, smoothed, label="Exponential Smoothing (α=0.2)",
                color="#D4A574", linewidth=1.2, linestyle='--')
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Smoothed Error")
    axes[2].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

