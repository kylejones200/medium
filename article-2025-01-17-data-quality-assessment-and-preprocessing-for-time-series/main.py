#!/usr/bin/env python3
"""
Data Quality Assessment and Preprocessing for Time Series

Main entry point for running data quality assessment and preprocessing.

Usage:
    python main.py
    python main.py --data-path data/timeseries.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import assess_data_quality, preprocess_time_series, plot_data_quality


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Data Quality Assessment and Preprocessing')
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
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        values = np.sin(np.arange(config['data']['n_periods']) / 10) + np.random.normal(0, 0.2, config['data']['n_periods'])
        values[np.random.choice(config['data']['n_periods'], 10)] = np.nan
        values[np.random.choice(config['data']['n_periods'], 5)] += 10
        df = pd.DataFrame({config['data']['value_column']: values}, index=dates)
    else:
        raise ValueError("No data source specified")
    
    print("Assessing data quality...")
    quality = assess_data_quality(df, config['data']['value_column'])
    
    print(f"\nData Quality Metrics:")
    print(f"Missing values: {quality['missing_values']} ({quality['missing_percentage']:.2f}%)")
    print(f"Duplicates: {quality['duplicates']}")
    print(f"Outliers: {quality['outliers']}")
    print(f"Data range: {quality['data_range']:.4f}")
    
    if config['preprocessing']['handle_missing'] or config['preprocessing']['handle_outliers']:
        print("\nPreprocessing data...")
        df_processed = preprocess_time_series(df, config['data']['value_column'])
        
        quality_processed = assess_data_quality(df_processed, config['data']['value_column'])
        print(f"\nAfter Preprocessing:")
        print(f"Missing values: {quality_processed['missing_values']}")
        print(f"Outliers: {quality_processed['outliers']}")
        
        plot_data_quality(df[config['data']['value_column']], 
                         df_processed[config['data']['value_column']],
                         "Data Quality: Before and After Preprocessing",
                         output_dir / 'data_quality_comparison.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

