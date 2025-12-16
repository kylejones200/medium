"""Core functions for treasury spread forecasting with Darts."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.metrics import mape, mse
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def fetch_fred_data(series_id: str, start_date: str = '2000-01-01', 
                   end_date: Optional[str] = None) -> TimeSeries:
    """Fetch FRED data using pandas_datareader and return as Darts TimeSeries."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    df = web.DataReader(series_id, 'fred', start=start_date, end=end_date)
    df = df.rename(columns={series_id: 'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce').ffill().dropna()
    df = df.sort_index()
    return TimeSeries.from_dataframe(df, value_cols='value')

def fit_auto_arima(series: TimeSeries, forecast_horizon: int = 30,
                  num_samples: int = 1000) -> Tuple[AutoARIMA, TimeSeries]:
    """Fit AutoARIMA model and return forecast."""
    if len(series) < forecast_horizon + 30:
        raise ValueError("Not enough data to forecast and evaluate.")
    
    model = AutoARIMA()
    model.fit(series[:-forecast_horizon])
    forecast = model.predict(n=forecast_horizon, num_samples=num_samples)
    return model, forecast

def evaluate_forecast(actual: TimeSeries, forecast: TimeSeries) -> Dict[str, float]:
    """Compute and return evaluation metrics."""
    mape_score = mape(actual, forecast)
    mse_score = mse(actual, forecast)
    return {
        'mape': mape_score,
        'mse': mse_score
    }

def forecast_to_dataframe(forecast: TimeSeries) -> pd.DataFrame:
    """Convert forecast to DataFrame."""
    return pd.DataFrame({
        'date': forecast.time_index,
        'forecast': forecast.values().flatten()
    })

def plot_forecast(series: TimeSeries, forecast: TimeSeries, output_path: Path,
                 title: str, metrics: Dict[str, float] = None, window: int = 365):
 """Plot forecast """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    display_window = min(window, len(series))
    series_display = series[-display_window:]
    
    series_display.plot(ax=ax, label="Actual", color="#4A90A4", linewidth=1.2)
    forecast.plot(ax=ax, label="Forecast", color="#D4A574", linewidth=1.2)
    
    title_text = title
    if metrics:
        title_text += f": MAPE = {metrics['mape']:.2f}%, MSE = {metrics['mse']:.4f}"
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

