#!/usr/bin/env python3
"""
Cross-Validation for Time Series Data

Main entry point for running time series cross-validation analysis.

Usage:
    python main.py --data-path data/my_data.csv
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.core import (
    TimeSeriesCV,
    RollingWindowCV,
    NestedTimeSeriesCV,
    BlockingTimeSeriesCV,
    TimeSeriesEvaluation,
    check_temporal_dependencies,
    check_data_leakage
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_example_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate example time series data for demonstration."""
    import numpy as np
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    np.random.seed(42)
    values = np.cumsum(np.random.randn(n_samples)) + 100
    return pd.DataFrame({
        'date': dates,
        'value': values
    })


def main():
    parser = argparse.ArgumentParser(description='Time Series Cross-Validation')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path, parse_dates=[config['data']['date_column']])
    else:
        print("No data file provided, generating example data...")
        df = generate_example_data()
    
    date_col = config['data']['date_column']
    target_col = config['data']['target_column']
    
    if config['cross_validation']['time_series_cv']['enabled']:
        print("Running Time Series Cross-Validation...")
        tscv = TimeSeriesCV(df, date_col, target_col)
        tscv.plot_time_series_split(
            config['cross_validation']['time_series_cv']['n_splits'],
            output_dir / 'time_series_cv_splits.png'
        )
    
    if config['analysis']['check_temporal_dependencies']:
        print("Checking temporal dependencies...")
        acf_vals = check_temporal_dependencies(
            df[target_col],
            config['analysis']['max_lag']
        )
        print(f"ACF values (first 5 lags): {acf_vals[:5]}")
    
    if config['analysis']['check_data_leakage']:
        print("Checking for data leakage...")
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(df):
            is_valid = check_data_leakage(train_idx, test_idx, df[date_col])
            print(f"Split valid (no leakage): {is_valid}")
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

