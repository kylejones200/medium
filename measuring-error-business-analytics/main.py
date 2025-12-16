#!/usr/bin/env python3
"""
Why Measuring Error Matters in Business Analytics (and How to Do It)

Main entry point for running error measurement analysis.
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
    parser = argparse.ArgumentParser(description='Measuring Error in Business Analytics')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        y_true = df['actual'].values if 'actual' in df.columns else df.iloc[:, 0].values
        y_pred = df['predicted'].values if 'predicted' in df.columns else df.iloc[:, 1].values
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        y_true = 100 + 20 * np.sin(np.arange(config['data']['n_samples']) / 10) + np.random.normal(0, 5, config['data']['n_samples'])
        y_pred = y_true + np.random.normal(0, config['model']['noise_level'] * y_true.std(), config['data']['n_samples'])
    else:
        raise ValueError("No data source specified")
    
    logging.info("Calculating error metrics...")
    metrics = calculate_error_metrics(y_true, y_pred)
    
    logging.error(" Metrics:")
    logging.info(f"MSE: {metrics['mse']:.4f}")
    logging.info(f"RMSE: {metrics['rmse']:.4f}")
    logging.info(f"MAE: {metrics['mae']:.4f}")
    logging.info(f"MAPE: {metrics['mape']:.2f}%")
    logging.info(f"R² Score: {metrics['r2']:.4f}")
    logging.info(f"Mean Error: {metrics['mean_error']:.4f}")
    logging.info(f"Std Error: {metrics['std_error']:.4f}")
    
    errors = y_true - y_pred
    plot_error_analysis(y_true, y_pred, errors, "Error Measurement Analysis",
                       output_dir / 'error_analysis.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

