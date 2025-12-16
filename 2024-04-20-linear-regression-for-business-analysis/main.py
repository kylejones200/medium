#!/usr/bin/env python3
"""
Linear Regression for Business Analysis

Main entry point for running linear regression analysis.
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
    parser = argparse.ArgumentParser(description='Linear Regression for Business Analysis')
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
        feature1 = np.random.normal(50, 15, config['data']['n_samples'])
        feature2 = np.random.normal(30, 10, config['data']['n_samples'])
        target = 2 * feature1 + 1.5 * feature2 + np.random.normal(0, 5, config['data']['n_samples'])
        
        df = pd.DataFrame({
            config['model']['feature_columns'][0]: feature1,
            config['model']['feature_columns'][1]: feature2,
            config['model']['target_column']: target
        })
    else:
        raise ValueError("No data source specified")
    
        X, y = prepare_regression_data(df, config['model']['feature_columns'], 
                                  config['model']['target_column'])
    
        model, metrics = fit_linear_regression(X, y)
    
    logging.info(f"\nRegression Metrics:")
    logging.info(f"R² Score: {metrics['r2']:.4f}")
    logging.info(f"RMSE: {metrics['rmse']:.4f}")
    logging.info(f"MAE: {metrics['mae']:.4f}")
    logging.info(f"\nCoefficients: {metrics['coefficients']}")
    logging.info(f"Intercept: {metrics['intercept']:.4f}")
    
    y_pred = model.predict(X)
    plot_regression_results(y, y_pred, "Linear Regression: Actual vs Predicted",
                           output_dir / 'regression_results.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

