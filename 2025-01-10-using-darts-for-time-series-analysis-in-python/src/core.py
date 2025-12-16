"""Core functions for time series forecasting with Darts."""

import warnings
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import requests

from darts import TimeSeries
from darts.models import ARIMA, ExponentialSmoothing, LightGBMModel, RNNModel, FFT, NBEATSModel
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.metrics import mae, mape, r2_score
from darts.utils.callbacks import TFMProgressBar
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

def fetch_fred_series(series_id: str, api_key: str, start: str = "2000-01-01") -> TimeSeries:
    """Fetch time series data from FRED API."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": datetime.now().strftime('%Y-%m-%d'),
    }
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception(f"FRED API error {r.status_code}")
    df = pd.DataFrame(r.json()["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").ffill()
    return TimeSeries.from_dataframe(df.sort_values("date"), "date", "value")

def load_data(data_path: Path = None, series_id: str = None, api_key: str = None) -> TimeSeries:
    """Load time series data from file or FRED API."""
    if data_path and data_path.exists():
        df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        return TimeSeries.from_dataframe(df, value_cols=['value'])
    elif series_id and api_key:
        return fetch_fred_series(series_id, api_key)
    else:
        raise ValueError("Either data_path or (series_id and api_key) must be provided")

def prepare_data(series: TimeSeries, split_date: str = "2020-01-01") -> Tuple[TimeSeries, TimeSeries, Scaler]:
    """Prepare data: fill missing values, split, and scale."""
    filler = MissingValuesFiller()
    series_filled = filler.transform(series)
    
    train, val = series_filled.split_before(pd.Timestamp(split_date))
    
    scaler = Scaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    
    return train_scaled, val_scaled, scaler

def fit_arima(train: TimeSeries, order: Tuple[int, int, int] = (1, 1, 1)) -> ARIMA:
    """Fit ARIMA model."""
    model = ARIMA(*order)
    model.fit(train)
    return model

def fit_exponential_smoothing(train: TimeSeries) -> ExponentialSmoothing:
    """Fit Exponential Smoothing model."""
    model = ExponentialSmoothing()
    model.fit(train)
    return model

def fit_lightgbm(train: TimeSeries, lags: int = 30) -> LightGBMModel:
    """Fit LightGBM model."""
    model = LightGBMModel(lags=lags)
    model.fit(train)
    return model

def fit_lstm(train: TimeSeries, val: TimeSeries = None, input_chunk_length: int = 30, 
             output_chunk_length: int = 7, n_epochs: int = 50) -> RNNModel:
    """Fit LSTM model."""
    model = RNNModel(model="LSTM", input_chunk_length=input_chunk_length, 
                     output_chunk_length=output_chunk_length, n_epochs=n_epochs)
    model.fit(train, val_series=val)
    return model

def fit_nbeats(train: TimeSeries, val: TimeSeries = None, input_chunk_length: int = 30,
               output_chunk_length: int = 7, n_epochs: int = 50, random_state: int = 42) -> NBEATSModel:
    """Fit NBEATS model."""
    torch_kwargs = {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }
    model = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length,
                       n_epochs=n_epochs, random_state=random_state, **torch_kwargs)
    model.fit(train, val_series=val)
    return model

def fit_fft(train: TimeSeries) -> FFT:
    """Fit FFT model."""
    model = FFT()
    model.fit(train)
    return model

def forecast_model(model: Any, n: int, scaler: Scaler = None) -> TimeSeries:
    """Generate forecast from model and optionally inverse transform."""
    forecast = model.predict(n)
    if scaler:
        forecast = scaler.inverse_transform(forecast)
    return forecast

def evaluate_forecast(actual: TimeSeries, forecast: TimeSeries) -> Dict[str, float]:
    """Evaluate forecast using multiple metrics."""
    return {
        'mae': mae(actual, forecast),
        'mape': mape(actual, forecast),
        'r2': r2_score(actual, forecast)
    }

def plot_forecast(series: TimeSeries, forecast: TimeSeries, title: str, output_path: Path,
                 metrics: Dict[str, float] = None):
 """Plot forecast vs actual """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    series.plot(ax=ax, label="Actual", color="#4A90A4", linewidth=1.2)
    forecast.plot(ax=ax, label="Forecast", color="#D4A574", linewidth=1.2)
    
    title_text = title
    if metrics:
        title_text += f": MAE = {metrics['mae']:.2f}, MAPE = {metrics['mape']:.2%}"
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

