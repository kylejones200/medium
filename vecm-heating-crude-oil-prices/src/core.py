"""Core functions for VECM modeling of heating oil and crude oil prices."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def fetch_data(symbols: list, start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Fetch historical prices from Yahoo Finance."""
    df = yf.download(symbols, start=start, end=end)['Close'].dropna()
    df.columns = ['HeatingOil', 'CrudeOil']
    return df

def test_stationarity(df: pd.DataFrame) -> Dict:
    """Perform ADF tests for stationarity."""
    results = {}
    for col in df.columns:
        result = adfuller(df[col])
        result_diff = adfuller(df[col].diff().dropna())
        results[col] = {
            'level': {'statistic': result[0], 'pvalue': result[1]},
            'diff': {'statistic': result_diff[0], 'pvalue': result_diff[1]}
        }
    return results

def test_cointegration(df: pd.DataFrame) -> Dict:
    """Perform Johansen cointegration test."""
    result = coint_johansen(df, det_order=0, k_ar_diff=1)
    return {
        'trace_statistic': result.lr1,
        'critical_values': result.cvt,
        'eigenvalues': result.eig
    }

def fit_vecm_model(df: pd.DataFrame, lags: int = 1, rank: int = 1):
    """Fit VECM model."""
    model = VECM(df, k_ar_diff=lags, coint_rank=rank)
    return model.fit()

def forecast_vecm(model, df: pd.DataFrame, steps: int = 12) -> pd.DataFrame:
    """Forecast using VECM model."""
    forecast = model.predict(steps=steps)
    future_idx = pd.date_range(df.index[-1], periods=steps, freq='ME')
    return pd.DataFrame(forecast, columns=df.columns, index=future_idx)

def plot_price_relationship(df: pd.DataFrame, forecast: pd.DataFrame = None,
                           title: str, output_path: Path):
 """Plot price relationship """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df['HeatingOil'], label="Heating Oil", color="#4A90A4", linewidth=1.2)
    ax.plot(df.index, df['CrudeOil'], label="Crude Oil", color="#D4A574", linewidth=1.2)
    
    if forecast is not None:
        ax.plot(forecast.index, forecast['HeatingOil'], linestyle='--',
               color="#4A90A4", linewidth=1.2, alpha=0.7, label="Forecast (Heating Oil)")
        ax.plot(forecast.index, forecast['CrudeOil'], linestyle='--',
               color="#D4A574", linewidth=1.2, alpha=0.7, label="Forecast (Crude Oil)")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

