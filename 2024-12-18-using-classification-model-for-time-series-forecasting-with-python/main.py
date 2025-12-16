#!/usr/bin/env python3
"""
Using Classification Model for Time Series Forecasting

Main entry point for running classification-based forecasting.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    create_classification_targets,
    create_lagged_features,
    train_classification_model,
)

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Using Classification Model for Time Series Forecasting')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        data = df.iloc[:, 0]
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        data = pd.Series(np.sin(np.arange(config['data']['n_periods']) / 10) + 
                        np.random.normal(0, 0.1, config['data']['n_periods']))
    else:
        raise ValueError("No data source specified")
    
        y_class, encoder = create_classification_targets(data, config['model']['n_bins'])
    
        X, y = create_lagged_features(data, config['model']['lag'])
    y_class_lagged = y_class[config['model']['lag']:]
    
    train_size = int(len(X) * config['model']['train_size'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_class_lagged[:train_size], y_class_lagged[train_size:]
    
        model = train_classification_model(X_train, y_train)
    
        y_pred = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"\nClassification Accuracy: {accuracy:.4f}")
    
    plot_classification_results(y_test, y_pred, "Classification-Based Forecasting",
                               output_dir / 'classification_forecast.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

