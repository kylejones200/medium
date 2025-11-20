#!/usr/bin/env python3
"""
Treasury Spread Forecasting with Darts

Main entry point for running AutoARIMA forecasting on treasury spread data.

Usage:
    python main.py
    python main.py --config custom_config.yaml
    python main.py --series-id T10Y2Y --forecast-horizon 30
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import (
    fetch_fred_data,
    fit_auto_arima,
    evaluate_forecast,
    forecast_to_dataframe,
    plot_forecast
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Treasury Spread Forecasting with Darts')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--series-id', type=str, default=None, help='FRED series ID')
    parser.add_argument('--forecast-horizon', type=int, default=None, help='Forecast horizon')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    series_id = args.series_id or config['data']['series_id']
    forecast_horizon = args.forecast_horizon or config['model']['forecast_horizon']
    
    try:
        logging.info("Fetching FRED data...")
        series = fetch_fred_data(
            series_id,
            config['data']['start_date'],
            config['data']['end_date']
        )
        
        logging.info("Fitting model and generating forecast...")
        model, forecast = fit_auto_arima(
            series,
            forecast_horizon,
            config['model']['num_samples']
        )
        
        if config['analysis']['print_summary']:
            logging.info("Forecast summary:")
            forecast_df = forecast_to_dataframe(forecast)
            print(forecast_df.head())
        
        metrics = None
        if config['analysis']['evaluate_forecast']:
            actual = series[-forecast_horizon:]
            metrics = evaluate_forecast(actual, forecast)
            logging.info("Evaluation Metrics:")
            print(f"MAPE: {metrics['mape']:.2f}%")
            print(f"MSE:  {metrics['mse']:.4f}")
        
        logging.info("Visualizing forecast...")
        title = f"{series_id} Treasury Spread Forecast (AutoARIMA)"
        plot_forecast(
            series,
            forecast,
            output_dir / 'arima_forecast.png',
            title,
            metrics,
            config['output']['display_window']
        )
        
        logging.info(f"Analysis complete. Figures saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

