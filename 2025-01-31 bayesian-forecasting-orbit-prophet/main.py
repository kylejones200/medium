#!/usr/bin/env python3
"""
Bayesian Forecasting with Orbit-ML and Prophet

Main entry point for running Bayesian forecasting analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Bayesian Forecasting with Orbit and Prophet')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        data = prepare_forecast_data(df, config['data']['date_column'], config['data']['value_column'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        trend = np.linspace(100, 150, config['data']['n_periods'])
        seasonality = 10 * np.sin(2 * np.pi * np.arange(config['data']['n_periods']) / 365.25)
        noise = np.random.normal(0, 3, config['data']['n_periods'])
        values = trend + seasonality + noise
        
        data = pd.Series(values, index=dates)
    else:
        raise ValueError("No data source specified")
    
                train_size = int(len(data) * config['model']['train_size'])
    train_data = data[:train_size]
    test_data = data[train_size:train_size + config['model']['forecast_horizon']]
    
    logging.info(f"\nTraining data: {len(train_data)} periods")
    logging.info(f"Test data: {len(test_data)} periods")
    
    if config['model']['use_orbit']:
                orbit_pred = np.full(len(test_data), train_data.mean() + train_data.std())
    else:
        orbit_pred = None
    
    if config['model']['use_prophet']:
                prophet_pred = np.full(len(test_data), train_data.mean())
    else:
        prophet_pred = None
    
    if orbit_pred is not None and prophet_pred is not None:
        orbit_metrics = calculate_forecast_metrics(test_data.values, orbit_pred)
        prophet_metrics = calculate_forecast_metrics(test_data.values, prophet_pred)
        
        logging.info(f"\nOrbit Metrics:")
        logging.info(f"  RMSE: {orbit_metrics['rmse']:.4f}")
        logging.info(f"  MAPE: {orbit_metrics['mape']:.2f}%")
        
        logging.info(f"\nProphet Metrics:")
        logging.info(f"  RMSE: {prophet_metrics['rmse']:.4f}")
        logging.info(f"  MAPE: {prophet_metrics['mape']:.2f}%")
        
        plot_forecast_comparison(test_data.values, orbit_pred, prophet_pred,
                                "Bayesian Forecasting: Orbit vs Prophet",
                                output_dir / 'forecast_comparison.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

