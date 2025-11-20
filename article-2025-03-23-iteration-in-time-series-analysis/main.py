#!/usr/bin/env python3
"""
Iteration in Time Series Analysis

Main entry point for running iterative time series analysis.

Usage:
    python main.py
    python main.py --data-path data/timeseries.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import iterative_forecast, calculate_iteration_metrics, plot_iteration_results


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Iteration in Time Series Analysis')
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
    
    print("Performing iterative forecasting...")
    forecasts = iterative_forecast(data, config['model']['n_iterations'], config['model']['horizon'])
    
    print("Calculating iteration metrics...")
    metrics = calculate_iteration_metrics(forecasts, data.values)
    for m in metrics:
        print(f"Iteration {m['iteration']}: MSE = {m['mse']:.4f}")
    
    plot_iteration_results(data, forecasts, "Iterative Time Series Forecasting",
                          output_dir / 'iteration_results.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

