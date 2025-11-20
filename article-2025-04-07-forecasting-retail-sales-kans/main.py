#!/usr/bin/env python3
"""
Forecasting Retail Sales with KANs

Main entry point for running retail sales forecasting.

Usage:
    python main.py
    python main.py --data-path data/sales.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    prepare_sales_data,
    create_lagged_features,
    calculate_forecast_metrics,
    plot_forecast
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Forecasting Retail Sales with KANs')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df = prepare_sales_data(df, config['data']['date_column'], config['data']['sales_column'])
        sales = df[config['data']['sales_column']]
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        sales = pd.Series(
            np.sin(np.arange(config['data']['n_periods']) / 30) * 100 + 500 + np.random.normal(0, 20, config['data']['n_periods']),
            index=dates
        )
    else:
        raise ValueError("No data source specified")
    
    print("Creating features...")
    X, y = create_lagged_features(sales, config['model']['lag'])
    
    train_size = int(len(X) * config['model']['train_size'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Note: KAN model implementation would go here")
    print("For demonstration, using simple baseline prediction...")
    y_pred = np.mean(X_test, axis=1)
    
    metrics = calculate_forecast_metrics(y_test, y_pred)
    print(f"\nForecast Metrics:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    
    plot_forecast(y_test, y_pred, "Retail Sales Forecast", output_dir / 'sales_forecast.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

