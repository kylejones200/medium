#!/usr/bin/env python3
"""
Time Series Manipulation with Pandas

Main entry point for running time series manipulation analysis.

Usage:
    python main.py
    python main.py --data-path data/timeseries.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import manipulate_time_series, resample_time_series, plot_time_series_manipulation


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Time Series Manipulation with Pandas')
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
        values = 100 + 10 * np.sin(np.arange(config['data']['n_periods']) / 30) + np.random.normal(0, 3, config['data']['n_periods'])
        df = pd.DataFrame({
            config['data']['date_column']: dates,
            config['data']['value_column']: values
        })
    else:
        raise ValueError("No data source specified")
    
    print("Manipulating time series...")
    df_manipulated = manipulate_time_series(df, config['data']['date_column'], 
                                           config['data']['value_column'])
    
    print(f"\nTime Series Statistics:")
    print(f"Mean: {df_manipulated[config['data']['value_column']].mean():.2f}")
    print(f"Std: {df_manipulated[config['data']['value_column']].std():.2f}")
    
    print(f"\nResampling to {config['manipulation']['resample_freq']}...")
    resampled = resample_time_series(df_manipulated[config['data']['value_column']],
                                    config['manipulation']['resample_freq'])
    print(f"Resampled length: {len(resampled)}")
    
    plot_time_series_manipulation(df_manipulated, config['data']['value_column'],
                                 "Time Series Manipulation", output_dir / 'manipulation.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

