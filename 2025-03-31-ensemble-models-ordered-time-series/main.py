#!/usr/bin/env python3
"""
Ensemble Models for Ordered Time Series

Main entry point for running ensemble model analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    create_ordered_features,
    train_ensemble_models,
    ensemble_predict,
)

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Ensemble Models for Ordered Time Series')
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
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        values = np.sin(np.arange(config['data']['n_periods']) / 10) + np.random.normal(0, 0.1, config['data']['n_periods'])
        data = pd.Series(values, index=dates)
    else:
        raise ValueError("No data source specified")
    
        features_df = create_ordered_features(data, config['model']['lag'])
    X = features_df.drop(columns=['target']).values
    y = features_df['target'].values
    
    train_size = int(len(X) * config['model']['train_size'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
        models = train_ensemble_models(X_train, y_train)
    
        individual_preds = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        individual_preds[name] = pred
    
    ensemble_pred = ensemble_predict(models, X_test, config['model']['ensemble_method'])
    
    from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"\nEnsemble RMSE: {np.sqrt(mean_squared_error(y_test, ensemble_pred)):.4f}")
    
    plot_ensemble_forecast(y_test, individual_preds, ensemble_pred,
                          "Ensemble Model Forecast", output_dir / 'ensemble_forecast.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

