#!/usr/bin/env python3
"""
Alternative Metrics for Predictive Maintenance

Main entry point for running alternative metrics analysis.

Usage:
    python main.py
    python main.py --data-path data/sensor_data.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    calculate_health_index,
    calculate_degradation_rate,
    calculate_remaining_useful_life,
    plot_health_metrics
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Alternative Metrics for Predictive Maintenance')
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
    
    print("Calculating health index...")
    health = calculate_health_index(df, config['model']['sensor_columns'])
    
    print("Calculating degradation rate...")
    degradation = calculate_degradation_rate(health, config['model']['degradation_window'])
    
    print("Calculating remaining useful life...")
    rul = calculate_remaining_useful_life(health, config['model']['health_threshold'])
    
    print(f"Mean RUL: {rul.mean():.2f}")
    print(f"Min RUL: {rul.min():.2f}")
    
    plot_health_metrics(health, degradation, rul, config['model']['health_threshold'],
                       output_dir / 'health_metrics.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

