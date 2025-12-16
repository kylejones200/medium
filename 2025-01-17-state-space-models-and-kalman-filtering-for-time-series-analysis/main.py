#!/usr/bin/env python3
"""
State Space Models and Kalman Filtering for Time Series

Main entry point for running Kalman filter analysis.
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
    parser = argparse.ArgumentParser(description='State Space Models and Kalman Filtering')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        y = df.iloc[:, 0].values
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        y = np.cumsum(np.random.normal(0, 1, config['data']['n_periods']))
    else:
        raise ValueError("No data source specified")
    
    F = np.array(config['model']['F'])
    H = np.array(config['model']['H'])
    Q = np.array(config['model']['Q'])
    R = np.array(config['model']['R'])
    x0 = np.zeros(config['model']['state_dim'])
    P0 = np.eye(config['model']['state_dim'])
    
        x_filtered, P = kalman_filter(y, F, H, Q, R, x0, P0)
    
    plot_kalman_filter(y, x_filtered, "Kalman Filter Results",
                      output_dir / 'kalman_filter.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

