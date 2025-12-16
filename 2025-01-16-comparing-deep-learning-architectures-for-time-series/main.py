#!/usr/bin/env python3
"""
Comparing Deep Learning Architectures for Time Series

Main entry point for comparing different deep learning architectures.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    prepare_data_for_lstm,
    calculate_model_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Comparing Deep Learning Architectures')
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
    
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(
        data, config['model']['lag'], config['model']['train_size']
    )
    
            results = {}
    for arch in config['model']['architectures']:
        y_pred = np.full(len(y_test), y_train.mean())
        y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        results[arch] = y_pred_inverse
        
        metrics = calculate_model_metrics(y_test, y_pred)
        logging.info(f"\n{arch} Metrics:")
        logging.info(f"  RMSE: {metrics['rmse']:.4f}")
        logging.info(f"  MAE: {metrics['mae']:.4f}")
    
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    plot_architecture_comparison(results, y_test_inverse,
                                "Deep Learning Architecture Comparison",
                                output_dir / 'architecture_comparison.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

