"""Core functions for ARAR (AutoRegressive AutoRegressive) algorithm forecasting."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_data(data_path: Path, date_column: str = "date", 
             value_column: str = "values") -> pd.Series:
    """Load time series data from CSV file."""
    data = pd.read_csv(data_path, parse_dates=[date_column], index_col=date_column)
    return data[value_column]

def apply_differencing(series: pd.Series) -> np.ndarray:
    """Apply first-order differencing to remove trend."""
    return np.diff(series)

def select_reduced_lags(acf_vals: np.ndarray, max_lag: int = 20,
                       strategy: str = "powers_of_2") -> List[int]:
    """Select reduced lag set based on strategy."""
    if strategy == "powers_of_2":
        lags = [1, 2, 4, 8, 16]
        return [lag for lag in lags if lag <= max_lag]
    elif strategy == "acf_threshold":
        threshold = 0.2
        significant_lags = np.where(np.abs(acf_vals[1:]) > threshold)[0] + 1
        return significant_lags.tolist()[:10]
    else:
        return [1, 2, 4, 8]

def fit_arar_model(differenced_data: np.ndarray, lags: List[int]):
    """Fit ARAR model using AutoReg with selected lags."""
    model = AutoReg(differenced_data, lags=lags, old_names=False)
    return model.fit()

def generate_arar_forecast(model, h: int, last_value: float) -> np.ndarray:
    """Generate h-step forecast using ARAR model."""
    future_forecast = model.predict(start=len(model.model.endog), 
                                   end=len(model.model.endog) + h - 1)
    y_forecast = np.cumsum(future_forecast) + last_value
    return y_forecast

def fit_arima_model(train: pd.Series, order: Tuple[int, int, int] = (2, 1, 2)):
    """Fit ARIMA model."""
    model = ARIMA(train, order=order)
    return model.fit()

def generate_arima_forecast(model, h: int) -> pd.Series:
    """Generate h-step forecast using ARIMA model."""
    return model.forecast(steps=h)

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Mean Absolute Percentage Error."""
    return mean_absolute_percentage_error(actual, predicted)

def plot_series_comparison(original: pd.Series, differenced: np.ndarray,
                          output_path: Path):
 """Plot original and differenced series """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(original.index, original.values, color="#4A90A4", linewidth=1.2)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    
    diff_index = original.index[1:]
    axes[1].plot(diff_index, differenced, color="#D4A574", linewidth=1.2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Differenced Value")
    
    plt.suptitle("ARAR Algorithm: Original and Differenced Series", 
                fontsize=12, y=0.98, color='0.2')
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_forecast_comparison(y: pd.Series, y_forecast_arar: np.ndarray,
                            forecast_index: pd.DatetimeIndex, h: int,
                            output_path: Path):
 """Plot forecast """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(y.index, y.values, label="Original Series", color="#4A90A4", linewidth=1.2)
    ax.plot(forecast_index, y_forecast_arar, label=f"{h}-Step Forecast", 
           color="#D4A574", linewidth=1.2, linestyle="--")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_arar_vs_arima(y: pd.Series, train: pd.Series, test: pd.Series,
                      y_forecast_arar: pd.Series, y_forecast_arima: pd.Series,
                      forecast_index: pd.DatetimeIndex, mape_arar: float,
                      mape_arima: float, output_path: Path):
 """Plot ARAR vs ARIMA forecast comparison """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(y.index, y.values, label="Historical Data", color="#4A90A4", linewidth=1.2)
    ax.plot(forecast_index, y_forecast_arar, label="ARAR Forecast", 
           color="#D4A574", linewidth=1.2, linestyle="--")
    ax.plot(forecast_index, y_forecast_arima, label="ARIMA Forecast", 
           color="#8B6F9E", linewidth=1.2, linestyle=":")
    
    title = f"ARAR vs ARIMA Forecasts: MAPE (ARAR) = {mape_arar:.4f}, MAPE (ARIMA) = {mape_arima:.4f}"
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

