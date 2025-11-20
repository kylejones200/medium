#!/usr/bin/env python3
"""
Nonparametric Regression with KNN and Kernel Smoothing

Main entry point for running nonparametric regression analysis.

Usage:
    python main.py
    python main.py --data-path data/temperatures.csv
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    load_temperature_data,
    create_lag_features,
    split_time_series,
    train_knn_regressor,
    apply_kernel_smoothing,
    plot_time_series,
    plot_knn_predictions,
    plot_kernel_smoothing
)
from sklearn.metrics import mean_squared_error


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Nonparametric Regression Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        print(f"Loading data from {args.data_path}...")
        data = pd.read_csv(args.data_path, parse_dates=['Date'], index_col='Date')
    else:
        print("Loading temperature data from URL...")
        data = load_temperature_data(config['data']['source_url'])
    
    plot_time_series(data, config['data']['target_column'], 
                    output_dir / 'raw_temperatures.png')
    
    data = create_lag_features(data, config['features']['max_lag'], 
                              config['data']['target_column'])
    
    train, test = split_time_series(data, config['data']['split_date'])
    X_train = train.drop(config['data']['target_column'], axis=1)
    y_train = train[config['data']['target_column']]
    X_test = test.drop(config['data']['target_column'], axis=1)
    y_test = test[config['data']['target_column']]
    
    print("Training k-NN regressor...")
    knn = train_knn_regressor(X_train, y_train, config['model']['knn']['n_neighbors'])
    y_pred_knn = knn.predict(X_test)
    
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    print(f'k-NN Mean Squared Error: {mse_knn:.2f}')
    
    plot_knn_predictions(y_test, y_pred_knn, output_dir / 'knn_predictions.png')
    
    print("Applying kernel smoothing...")
    y_pred_kernel = apply_kernel_smoothing(data, config['data']['target_column'],
                                           config['model']['kernel']['bandwidth'])
    
    plot_kernel_smoothing(data, y_pred_kernel, config['data']['target_column'],
                         output_dir / 'kernel_smoothing.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    import pandas as pd
    main()

