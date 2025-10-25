import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.metrics import mape, mse
import pandas_datareader.data as web

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
np.random.seed(42)

def fetch_fred_data(series_id, start_date='2000-01-01', end_date=None):
    """Fetch FRED data using pandas_datareader and return as Darts TimeSeries."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    df = web.DataReader(series_id, 'fred', start=start_date, end=end_date)
    df = df.rename(columns={series_id: 'value'})
    df['value'] = pd.to_numeric(df['value'], errors='coerce').ffill().dropna()
    df = df.sort_index()
    return TimeSeries.from_dataframe(df, value_cols='value')

def build_and_forecast(series, forecast_horizon=30, num_samples=1000):
    """Fit AutoARIMA model and return forecast."""
    if len(series) < forecast_horizon + 30:
        raise ValueError("Not enough data to forecast and evaluate.")
    model = AutoARIMA()
    model.fit(series[:-forecast_horizon])  # Leave last points out for evaluation
    forecast = model.predict(n=forecast_horizon, num_samples=num_samples)
    return forecast

def visualize_forecast(series, forecast, title, filename):
    """Plot and save forecast visualization."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    window = min(365, len(series))
    plt.figure(figsize=(12, 6))
    series[-window:].plot(label="Actual", color='blue')
    forecast.plot(label="Forecast", color='red')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(filename)
    plt.show()

def print_forecast_summary(forecast):
    """Convert forecast to DataFrame and print head."""
    forecast_df = pd.DataFrame({
        'date': forecast.time_index,
        'forecast': forecast.values().flatten()
    })
    print(forecast_df.head())
    return forecast_df

def evaluate_forecast(actual, forecast):
    """Compute and print evaluation metrics."""
    mape_score = mape(actual, forecast)
    mse_score = mse(actual, forecast)
    print(f"\nEvaluation Metrics:")
    print(f"MAPE: {mape_score:.2f}%")
    print(f"MSE:  {mse_score:.4f}")
    return mape_score, mse_score

if __name__ == "__main__":
    series_id = "T10Y2Y"
    forecast_horizon = 30
    output_file = "outputs/ARIMA_Forecast.png"

    try:
        logging.info("Fetching FRED data...")
        series = fetch_fred_data(series_id)

        logging.info("Fitting model and generating forecast...")
        forecast = build_and_forecast(series, forecast_horizon)

        logging.info("Visualizing forecast...")
        visualize_forecast(series, forecast,
                           "10Y Minus 2Y Treasury Spread Forecast (AutoARIMA)",
                           output_file)

        logging.info("Forecast summary:")
        forecast_df = print_forecast_summary(forecast)

        # Evaluate
        actual = series[-forecast_horizon:]
        logging.info("Evaluating forecast...")
        evaluate_forecast(actual, forecast)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
