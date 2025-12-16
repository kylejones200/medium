#!/usr/bin/env python3
"""
Mathematical Foundations of Time Series Analysis

Main entry point for running mathematical foundations analysis.
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
    parser = argparse.ArgumentParser(description='Mathematical Foundations of Time Series')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df[config['data']['date_column']] = pd.to_datetime(df[config['data']['date_column']])
        df = df.set_index(config['data']['date_column'])
        series = df[config['data']['value_column']]
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        trend = np.linspace(100, 120, config['data']['n_periods'])
        noise = np.random.normal(0, 5, config['data']['n_periods'])
        series = pd.Series(trend + noise, index=dates)
    else:
        raise ValueError("No data source specified")
    
        properties = calculate_statistical_properties(series)
    
    logging.info(f"\nStatistical Properties:")
    logging.info(f"Mean: {properties['mean']:.4f}")
    logging.info(f"Variance: {properties['variance']:.4f}")
    logging.info(f"Standard Deviation: {properties['std']:.4f}")
    logging.info(f"Skewness: {properties['skewness']:.4f}")
    logging.info(f"Kurtosis: {properties['kurtosis']:.4f}")
    logging.info(f"Autocorrelation (lag 1): {properties['autocorr_lag1']:.4f}")
    
    if config['analysis']['calculate_acf']:
        logging.info(f"\nCalculating autocorrelation function...")
        acf = calculate_autocorrelation(series, config['analysis']['max_lag'])
        
        plot_mathematical_properties(series, acf, "Mathematical Foundations of Time Series",
                                    output_dir / 'mathematical_properties.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

