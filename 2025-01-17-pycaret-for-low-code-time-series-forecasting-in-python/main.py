#!/usr/bin/env python3
"""
PyCaret for Low-Code Time Series Forecasting

Main entry point for running PyCaret time series forecasting.
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
    parser = argparse.ArgumentParser(description='PyCaret for Low-Code Time Series Forecasting')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        data = prepare_time_series_data(df, config['data']['date_column'], config['data']['value_column'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        values = np.sin(np.arange(config['data']['n_periods']) / 10) + np.random.normal(0, 0.1, config['data']['n_periods'])
        data = pd.Series(values, index=dates)
    else:
        raise ValueError("No data source specified")
    
                                train_size = int(len(data) * config['pycaret']['train_size'])
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    logging.info(f"\nTraining data: {len(train_data)} periods")
    logging.info(f"Test data: {len(test_data)} periods")
    
    y_pred = np.full(len(test_data), train_data.mean())
    metrics = calculate_forecast_metrics(test_data.values, y_pred)
    logging.info(f"\nBaseline Metrics:")
    logging.info(f"  RMSE: {metrics['rmse']:.4f}")
    logging.info(f"  MAE: {metrics['mae']:.4f}")
    
    plot_pycaret_forecast(test_data.values, y_pred, "PyCaret Time Series Forecast",
                         output_dir / 'pycaret_forecast.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

