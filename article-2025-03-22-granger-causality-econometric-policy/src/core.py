"""Core functions for Granger Causality testing."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def fetch_fred_data(series: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from FRED using pandas_datareader."""
    from pandas_datareader import data as web
    
    dfs = []
    for s in series:
        df = web.DataReader(s, 'fred', start_date, end_date)
        dfs.append(df)
    
    return pd.concat(dfs, axis=1)


def test_stationarity(series: pd.Series, name: str) -> Dict[str, float]:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] <= 0.05
    }


def apply_differencing(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply first-order differencing to specified columns."""
    df = df.copy()
    for col in columns:
        df[f'{col}_diff'] = df[col].diff()
    return df


def run_granger_test(df: pd.DataFrame, y_col: str, x_col: str, maxlag: int = 4) -> Dict:
    """Run Granger causality test."""
    test_data = df[[y_col, x_col]].dropna()
    return grangercausalitytests(test_data, maxlag=maxlag, verbose=False)


def plot_time_series(df: pd.DataFrame, col1: str, col2: str,
                    label1: str, label2: str, output_path: Path):
    """Plot two time series with dual y-axes using Tufte style."""
    setup_tufte_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df['date'], df[col1], color="#4A90A4", linewidth=1.2)
    ax1.set_xlabel('Year')
    ax1.set_ylabel(label1, color="#4A90A4")
    ax1.tick_params(axis='y', labelcolor="#4A90A4")
    
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df[col2], color="#D4A574", linewidth=1.2)
    ax2.set_ylabel(label2, color="#D4A574")
    ax2.tick_params(axis='y', labelcolor="#D4A574")
    
    apply_tufte_style(ax1, title=f'{label1} and {label2} Over Time')
    
    save_tufte_figure(output_path)

