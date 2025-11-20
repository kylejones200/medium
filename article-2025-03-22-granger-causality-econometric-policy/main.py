#!/usr/bin/env python3
"""
Granger Causality Testing for Econometric Policy Analysis

Main entry point for running Granger causality tests.

Usage:
    python main.py
    python main.py --data-path data/unemployment_spending.csv
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.core import (
    fetch_fred_data,
    test_stationarity,
    apply_differencing,
    run_granger_test,
    plot_time_series
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Granger Causality Testing')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        print(f"Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path, parse_dates=['date'])
    else:
        print("Fetching data from FRED...")
        df = fetch_fred_data(
            config['data']['series'],
            config['data']['start_date'],
            config['data']['end_date']
        )
        df = df.reset_index().rename(columns={'DATE': 'date'})
        
        for old_col, new_col in config['data']['column_mapping'].items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        if config['data']['save_csv']:
            csv_path = Path(config['data']['csv_path'])
            csv_path.parent.mkdir(exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"Data saved to {csv_path}")
    
    print("\nADF Stationarity Tests (Original Series):")
    for col in ['unemployment_rate', 'consumer_spending']:
        if col in df.columns:
            result = test_stationarity(df[col], col)
            print(f"{col} ADF Statistic: {result['adf_statistic']:.3f}, "
                 f"p-value: {result['p_value']:.3f}, "
                 f"Stationary: {result['is_stationary']}")
    
    df = apply_differencing(df, ['unemployment_rate', 'consumer_spending'])
    
    print("\nADF Stationarity Tests (Differenced Series):")
    for col in ['unemployment_rate_diff', 'consumer_spending_diff']:
        if col in df.columns:
            result = test_stationarity(df[col], col)
            print(f"{col} ADF Statistic: {result['adf_statistic']:.3f}, "
                 f"p-value: {result['p_value']:.3f}, "
                 f"Stationary: {result['is_stationary']}")
    
    plot_time_series(df, 'unemployment_rate', 'consumer_spending',
                    'Unemployment Rate (%)', 'Consumer Spending (Billions)',
                    output_dir / 'unemployment_consumer_spending.png')
    
    print("\nGranger Causality Tests:")
    for test in config['analysis']['test_directions']:
        print(f"\n{test['description']}")
        try:
            result = run_granger_test(df, test['y'], test['x'], config['analysis']['maxlag'])
            print("Test completed. Check results above.")
        except Exception as e:
            print(f"Error running test: {e}")
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

