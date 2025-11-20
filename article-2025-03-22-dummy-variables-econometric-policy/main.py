#!/usr/bin/env python3
"""
Dummy Variables in Econometric Policy Analysis

Main entry point for running dummy variable analysis.

Usage:
    python main.py
    python main.py --data-path data/policy_data.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import create_dummy_variables, fit_policy_regression, plot_policy_effect


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Dummy Variables in Econometric Policy Analysis')
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
        policy_date = pd.to_datetime(config['policy']['policy_date'])
        values = np.random.normal(50, 5, config['data']['n_periods'])
        policy_idx = dates.get_loc(policy_date) if policy_date in dates else len(dates) // 2
        values[policy_idx:] += config['policy']['effect_size']
        df = pd.DataFrame({
            config['data']['date_column']: dates,
            config['data']['value_column']: values
        })
    else:
        raise ValueError("No data source specified")
    
    print("Creating dummy variables...")
    df = create_dummy_variables(df, config['data']['date_column'], config['policy']['policy_date'])
    
    print("Fitting policy regression...")
    x_cols = ['policy_dummy']
    if config['model']['include_trend']:
        df['trend'] = np.arange(len(df))
        x_cols.append('trend')
    
    results = fit_policy_regression(df, config['data']['value_column'], x_cols)
    print("\nRegression Results:")
    print(results.summary())
    
    plot_policy_effect(df, config['data']['value_column'], config['data']['date_column'],
                      config['policy']['policy_date'], "Policy Effect Analysis",
                      output_dir / 'policy_effect.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

