#!/usr/bin/env python3
"""
Time Series Forecasting with Darts

Main entry point for running forecasting models.

Usage:
    python main.py
    python main.py --config custom_config.yaml
    python main.py --data-path data/my_data.csv
"""

import argparse
import yaml
import os
from pathlib import Path
from src.core import (
    load_data,
    prepare_data,
    fit_arima,
    fit_exponential_smoothing,
    fit_lightgbm,
    fit_lstm,
    fit_nbeats,
    fit_fft,
    forecast_model,
    evaluate_forecast,
    plot_forecast
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting with Darts')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--api-key', type=str, default=None, help='FRED API key')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_path = args.data_path if args.data_path else (
        Path(config['data']['source']) if config['data']['source'] else None
    )
    
    api_key = args.api_key or config['data']['api_key'] or os.getenv('FRED_API_KEY')
    series_id = config['data']['series_id']
    
    if data_path:
        series = load_data(data_path=data_path)
    elif api_key and series_id:
        series = load_data(series_id=series_id, api_key=api_key)
    else:
        raise ValueError("Must provide either --data-path or --api-key (or set in config)")
    
    train, val, scaler = prepare_data(series, config['data']['split_date'])
    
    print(f"Training: {len(train)} observations")
    print(f"Validation: {len(val)} observations")
    
    models_config = config['models']
    
    if models_config['arima']['enabled']:
        print("\nFitting ARIMA...")
        model = fit_arima(train, tuple(models_config['arima']['order']))
        forecast = forecast_model(model, len(val), scaler)
        metrics = evaluate_forecast(val, forecast)
        print(f"ARIMA - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2%}")
        plot_forecast(series, forecast, "ARIMA Forecast", output_dir / 'arima_forecast.png', metrics)
    
    if models_config['exponential_smoothing']['enabled']:
        print("\nFitting Exponential Smoothing...")
        model = fit_exponential_smoothing(train)
        forecast = forecast_model(model, len(val), scaler)
        metrics = evaluate_forecast(val, forecast)
        print(f"Exponential Smoothing - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2%}")
        plot_forecast(series, forecast, "Exponential Smoothing Forecast", 
                    output_dir / 'exponential_smoothing_forecast.png', metrics)
    
    if models_config['lightgbm']['enabled']:
        print("\nFitting LightGBM...")
        model = fit_lightgbm(train, models_config['lightgbm']['lags'])
        forecast = forecast_model(model, len(val), scaler)
        metrics = evaluate_forecast(val, forecast)
        print(f"LightGBM - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2%}")
        plot_forecast(series, forecast, "LightGBM Forecast", 
                    output_dir / 'lightgbm_forecast.png', metrics)
    
    if models_config['lstm']['enabled']:
        print("\nFitting LSTM...")
        model = fit_lstm(train, val, 
                        models_config['lstm']['input_chunk_length'],
                        models_config['lstm']['output_chunk_length'],
                        models_config['lstm']['n_epochs'])
        forecast = forecast_model(model, len(val), scaler)
        metrics = evaluate_forecast(val, forecast)
        print(f"LSTM - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2%}")
        plot_forecast(series, forecast, "LSTM Forecast", 
                    output_dir / 'lstm_forecast.png', metrics)
    
    if models_config['nbeats']['enabled']:
        print("\nFitting NBEATS...")
        model = fit_nbeats(train, val,
                          models_config['nbeats']['input_chunk_length'],
                          models_config['nbeats']['output_chunk_length'],
                          models_config['nbeats']['n_epochs'],
                          models_config['nbeats']['random_state'])
        forecast = forecast_model(model, len(val), scaler)
        metrics = evaluate_forecast(val, forecast)
        print(f"NBEATS - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2%}")
        plot_forecast(series, forecast, "NBEATS Forecast", 
                    output_dir / 'nbeats_forecast.png', metrics)
    
    if models_config['fft']['enabled']:
        print("\nFitting FFT...")
        model = fit_fft(train)
        forecast = forecast_model(model, len(val), scaler)
        metrics = evaluate_forecast(val, forecast)
        print(f"FFT - MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2%}")
        plot_forecast(series, forecast, "FFT Forecast", 
                    output_dir / 'fft_forecast.png', metrics)
    
    print(f"\nForecasting complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

