#!/usr/bin/env python3
"""
Getting to Know Pandas for Data Analytics

Main entry point for running Pandas data analytics.

Usage:
    python main.py
    python main.py --data-path data/dataset.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import perform_data_operations, analyze_dataframe, plot_dataframe_comparison


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Getting to Know Pandas for Data Analytics')
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
        df = pd.DataFrame({
            'id': range(config['data']['n_rows']),
            'category': np.random.choice(['A', 'B', 'C', 'D'], config['data']['n_rows']),
            'value': np.random.normal(50, 10, config['data']['n_rows']),
            'score': np.random.uniform(0, 100, config['data']['n_rows'])
        })
    else:
        raise ValueError("No data source specified")
    
    print("Original DataFrame:")
    print(df.head())
    
    analysis = analyze_dataframe(df)
    print(f"\nNumeric columns: {analysis['numeric_columns']}")
    print(f"Categorical columns: {analysis['categorical_columns']}")
    
    print("\nPerforming operations...")
    df_processed = perform_data_operations(df, config['analysis']['operations'])
    
    print("\nProcessed DataFrame:")
    print(df_processed.head())
    
    if config['analysis']['operations'] and len(df.select_dtypes(include=[np.number]).columns) > 0:
        col = df.select_dtypes(include=[np.number]).columns[0]
        plot_dataframe_comparison(df, df_processed, col, "Data Before and After Processing",
                                 output_dir / 'data_comparison.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

