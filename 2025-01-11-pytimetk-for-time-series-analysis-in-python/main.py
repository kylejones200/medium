#!/usr/bin/env python3
"""
PyTimeTK for Time Series Analysis

Main entry point for running PyTimeTK time series analysis.
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
    parser = argparse.ArgumentParser(description='PyTimeTK for Time Series Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        series = prepare_time_series_data(df, config['data']['date_column'], config['data']['value_column'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        values = np.sin(np.arange(config['data']['n_periods']) / 10) + np.random.normal(0, 0.1, config['data']['n_periods'])
        series = pd.Series(values, index=dates)
    else:
        raise ValueError("No data source specified")
    
                        features = analyze_time_series_features(series)
    logging.info(f"\nTime Series Features:")
    logging.info(f"Length: {features['length']}")
    logging.info(f"Mean: {features['mean']:.4f}")
    logging.info(f"Volatility: {features['volatility']:.4f}")
    logging.info(f"Trend: {features['trend']}")
    
    plot_pytimetk_analysis(series, "PyTimeTK Time Series Analysis",
                          output_dir / 'pytimetk_analysis.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

