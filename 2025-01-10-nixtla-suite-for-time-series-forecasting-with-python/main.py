#!/usr/bin/env python3
"""
Nixtla Suite for Time Series Forecasting

Main entry point for running StatsForecast forecasting.
"""

import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_synthetic_data,
    split_data,
    fit_statsforecast,
    generate_forecast,
    calculate_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Nixtla StatsForecast Forecasting')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
        df = generate_synthetic_data(
        config['data']['start_date'],
        config['data']['periods'],
        config['data']['frequency'],
        config['data']['seed']
    )
    
    train, hold_out = split_data(df, config['forecast']['hold_out_hours'])
    
        sf = fit_statsforecast(
        train,
        config['model']['season_length'],
        config['model']['freq'],
        config['model']['n_jobs']
    )
    
        forecasts = generate_forecast(sf, len(hold_out))
    forecasts['ds'] = hold_out['ds'].values
    
    actual = hold_out['y'].values
    predicted = forecasts['AutoARIMA'].values
    metrics = calculate_metrics(actual, predicted)
    
    logging.info(f"MSE: {metrics['mse']:.2f}")
    logging.info(f"RMSE: {metrics['rmse']:.2f}")
    logging.info(f"MAE: {metrics['mae']:.2f}")
    
    plot_forecast(df, hold_out, forecasts, output_dir / 'autarima_forecast.png',
                 "AutoARIMA", metrics)
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

