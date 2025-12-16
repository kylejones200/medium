"""Core functions for VAR, IRF, and FEVD analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def fetch_yfinance_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch data from Yahoo Finance."""
    df = yf.download(tickers, start=start, end=end)['Close'].dropna()
    return df

def compute_log_returns(df: pd.DataFrame, resample_freq: str = 'M') -> pd.DataFrame:
    """Resample to specified frequency and compute log returns."""
    monthly_data = df.resample(resample_freq).mean()
    log_returns = np.log(monthly_data).diff().dropna()
    return log_returns

def test_stationarity(series: pd.Series, name: str) -> Dict[str, any]:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] <= 0.05
    }

def fit_var_model(data: pd.DataFrame, max_lag: int = 12) -> Tuple[VAR, any]:
    """Fit VAR model with automatic lag selection."""
    model = VAR(data)
    lag_selection = model.select_order(max_lag)
    selected_lag = lag_selection.aic
    fitted_model = model.fit(selected_lag)
    return model, fitted_model

def plot_cumulative_irf(irf, fitted_model, shock_var: str, response_vars: list,
                        periods: int, output_path: Path):
 """Plot cumulative impulse response functions """
    fig, axes = plt.subplots(len(response_vars), 1, figsize=(10, 4 * len(response_vars)), sharex=True)
    
    if len(response_vars) == 1:
        axes = [axes]
    
    shock_idx = fitted_model.names.index(shock_var)
    
    for i, var in enumerate(response_vars):
        var_idx = fitted_model.names.index(var)
        irf_plot = irf.cum_effects[:, shock_idx, var_idx]
        
        axes[i].plot(range(periods + 1), irf_plot, color="#4A90A4", linewidth=1.2)
        axes[i].set_ylabel("Response")
        axes[i].legend([f'{shock_var} → {var}'], loc='best')
    
    axes[-1].set_xlabel("Months")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_irf(irf, output_path: Path):
 """Plot standard IRF """
    irf.plot(orth=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_fevd(fevd, output_path: Path):
 """Plot FEVD """
    fevd.plot()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

