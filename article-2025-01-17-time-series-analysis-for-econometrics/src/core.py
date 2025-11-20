"""Core functions for time series analysis in econometrics."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def perform_adf_test(series: pd.Series) -> dict:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series)
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def perform_cointegration_test(series1: pd.Series, series2: pd.Series) -> dict:
    """Perform Engle-Granger cointegration test."""
    result = coint(series1, series2)
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[2],
        'is_cointegrated': result[1] < 0.05
    }


def fit_var_model(data: pd.DataFrame, maxlags: int = 4) -> VAR:
    """Fit Vector Autoregression model."""
    model = VAR(data)
    return model.fit(maxlags=maxlags)


def plot_time_series(data: pd.DataFrame, title: str, output_path: Path):
    """Plot time series with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for col in data.columns:
        ax.plot(data.index if hasattr(data.index, '__len__') else range(len(data)),
               data[col], label=col, linewidth=1.2)
    
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    save_tufte_figure(output_path)

