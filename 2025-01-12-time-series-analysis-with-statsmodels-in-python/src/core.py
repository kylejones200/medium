"""Core functions for time series analysis with statsmodels."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_synthetic_data(
    n_samples: int = 200,
    start_date: str = "2023-01-01",
    frequency: str = "D",
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic time series data with trend and seasonality."""
    np.random.seed(seed)
    time = pd.date_range(start=start_date, periods=n_samples, freq=frequency)
    trend = np.linspace(10, 50, n_samples)
    seasonality = 10 * np.sin(np.linspace(0, 2 * np.pi, n_samples))
    noise = np.random.normal(0, 2, n_samples)
    data = trend + seasonality + noise
    
    df = pd.DataFrame({"date": time, "value": data})
    df.set_index("date", inplace=True)
    return df

def load_data(data_path: Path = None) -> pd.DataFrame:
    """Load time series data from file or generate synthetic data."""
    if data_path and data_path.exists():
        df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    else:
        df = generate_synthetic_data()
    return df

def split_data(df: pd.DataFrame, hold_out_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and hold-out sets."""
    train = df.iloc[:-hold_out_days]
    hold_out = df.iloc[-hold_out_days:]
    return train, hold_out

def test_stationarity(series: pd.Series) -> Dict[str, Any]:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series)
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] <= 0.05,
        'critical_values': result[4]
    }

def decompose_series(series: pd.Series, model: str = "additive", period: int = 30):
    """Decompose time series into trend, seasonal, and residual components."""
    return seasonal_decompose(series, model=model, period=period)

def fit_arima(train_data: pd.Series, order: Tuple[int, int, int] = (2, 1, 2), freq: str = "D"):
    """Fit ARIMA model to training data."""
    model = ARIMA(train_data, order=order, freq=freq)
    return model.fit()

def forecast_arima(model, steps: int, index: pd.DatetimeIndex):
    """Generate ARIMA forecast."""
    forecast = model.get_forecast(steps=steps)
    return {
        'mean': forecast.predicted_mean,
        'conf_int': forecast.conf_int(),
        'index': index
    }

def fit_holt_winters(train_data: pd.Series, seasonal_periods: int = 30):
    """Fit Holt-Winters exponential smoothing model."""
    model = ExponentialSmoothing(
        train_data,
        seasonal="add",
        seasonal_periods=seasonal_periods
    )
    return model.fit()

def forecast_holt_winters(model, steps: int):
    """Generate Holt-Winters forecast."""
    return model.forecast(steps=steps)

def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Mean Absolute Percentage Error."""
    return mean_absolute_percentage_error(actual, predicted)

def plot_time_series(
    df: pd.DataFrame,
    train: pd.DataFrame,
    hold_out: pd.DataFrame,
    output_path: Path
):
    """Plot time series with training and hold-out sets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df["value"], label="Full Dataset", color="#4A90A4", linewidth=1.2)
    ax.plot(hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="#8B6F9E", linewidth=1.2)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_arima_forecast(
    train: pd.DataFrame,
    hold_out: pd.DataFrame,
    forecast: Dict[str, Any],
    mape: float,
    output_path: Path
):
    """Plot ARIMA forecast results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train.index, train["value"], label="Training Data", color="#4A90A4", linewidth=1.2)
    ax.plot(hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="#8B6F9E", linewidth=1.2)
    ax.plot(forecast['index'], forecast['mean'], label="ARIMA Forecast", color="#D4A574", linewidth=1.2)
    ax.fill_between(
        forecast['index'],
        forecast['conf_int'].iloc[:, 0],
        forecast['conf_int'].iloc[:, 1],
        color="#D4A574",
        alpha=0.12,
        label="95% Confidence Interval"
    )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_holt_winters_forecast(
    train: pd.DataFrame,
    hold_out: pd.DataFrame,
    forecast: pd.Series,
    mape: float,
    output_path: Path
):
    """Plot Holt-Winters forecast results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train.index, train["value"], label="Training Data", color="#4A90A4", linewidth=1.2)
    ax.plot(hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="#8B6F9E", linewidth=1.2)
    ax.plot(hold_out.index, forecast, label="Holt-Winters Forecast", color="#D4A574", linewidth=1.2)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_decomposition(decomposition, output_path: Path):
    """Plot time series decomposition."""
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    
    for ax in fig.axes:
    
    plt.suptitle("Time Series Decomposition: Trend, Seasonal, and Residual Components", 
                 fontsize=12, y=0.98, color='0.2')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_acf_pacf(series: pd.Series, lags: int = 30, output_path: Path = None):
    """Plot ACF and PACF."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1])
    
    for ax in axes:
    
    plt.suptitle("Autocorrelation and Partial Autocorrelation Functions", 
                 fontsize=12, y=0.98, color='0.2')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_residuals_diagnostics(model, output_path: Path):
    """Plot ARIMA residual diagnostics."""
    
    for ax in fig.axes:
    
    plt.suptitle("ARIMA Model Residual Diagnostics", fontsize=12, y=0.98, color='0.2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

