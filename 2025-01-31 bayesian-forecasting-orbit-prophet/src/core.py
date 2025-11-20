"""Core functions for Bayesian forecasting with Orbit and Prophet."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def prepare_forecast_data(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """Prepare data for forecasting."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    return df[value_col]


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate forecast error metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'mape': mape
    }


def plot_forecast_comparison(actual: np.ndarray, orbit_pred: np.ndarray,
                            prophet_pred: np.ndarray, title: str, output_path: Path):
    """Plot forecast comparison with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = np.arange(len(actual))
    ax.plot(time, actual, label="Actual", color="#4A90A4", linewidth=1.2)
    ax.plot(time, orbit_pred, label="Orbit", color="#D4A574", linewidth=1.2, linestyle='--')
    ax.plot(time, prophet_pred, label="Prophet", color="#8B6F9E", linewidth=1.2, linestyle='--')
    
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    save_tufte_figure(output_path)

