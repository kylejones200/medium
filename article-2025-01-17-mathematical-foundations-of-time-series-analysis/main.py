#!/usr/bin/env python3
"""
Mathematical Foundations of Time Series Analysis

Main entry point for running mathematical foundations analysis.

Usage:
    python main.py
    python main.py --data-path data/timeseries.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import calculate_statistical_properties, calculate_autocorrelation, plot_mathematical_properties


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
    
    print("Calculating statistical properties...")
    properties = calculate_statistical_properties(series)
    
    print(f"\nStatistical Properties:")
    print(f"Mean: {properties['mean']:.4f}")
    print(f"Variance: {properties['variance']:.4f}")
    print(f"Standard Deviation: {properties['std']:.4f}")
    print(f"Skewness: {properties['skewness']:.4f}")
    print(f"Kurtosis: {properties['kurtosis']:.4f}")
    print(f"Autocorrelation (lag 1): {properties['autocorr_lag1']:.4f}")
    
    if config['analysis']['calculate_acf']:
        print(f"\nCalculating autocorrelation function...")
        acf = calculate_autocorrelation(series, config['analysis']['max_lag'])
        
        plot_mathematical_properties(series, acf, "Mathematical Foundations of Time Series",
                                    output_dir / 'mathematical_properties.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

