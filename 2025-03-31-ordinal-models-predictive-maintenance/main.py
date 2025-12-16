#!/usr/bin/env python3
"""
Ordinal Models for Predictive Maintenance

Main entry point for running ordinal model analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    create_ordinal_targets,
    create_maintenance_features,
    train_ordinal_model,
)

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Ordinal Models for Predictive Maintenance')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        sensor1 = 1.0 - np.linspace(0, 0.8, config['data']['n_periods']) + np.random.normal(0, 0.05, config['data']['n_periods'])
        sensor2 = 1.0 - np.linspace(0, 0.7, config['data']['n_periods']) + np.random.normal(0, 0.05, config['data']['n_periods'])
        df = pd.DataFrame({
            'date': dates,
            'sensor_1': sensor1,
            'sensor_2': sensor2
        })
    else:
        raise ValueError("No data source specified")
    
        health_index = (df[config['model']['sensor_columns']].mean(axis=1))
    y_ordinal, encoder = create_ordinal_targets(health_index, config['model']['n_levels'])
    
        X = create_maintenance_features(df, config['model']['sensor_columns'])
    
    train_size = int(len(X) * config['model']['train_size'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_ordinal[:train_size], y_ordinal[train_size:]
    
        model = train_ordinal_model(X_train, y_train)
    
        y_pred = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"\nClassification Accuracy: {accuracy:.4f}")
    
    levels = config['model']['degradation_levels']
    plot_ordinal_predictions(y_test, y_pred, levels,
                           "Ordinal Model: Degradation Level Prediction",
                           output_dir / 'ordinal_predictions.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

