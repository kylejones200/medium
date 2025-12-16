#!/usr/bin/env python3
"""
Basic Data Analysis using Pandas Library

Main entry point for running basic data analysis.
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
    parser = argparse.ArgumentParser(description='Basic Data Analysis using Pandas')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = load_and_explore_data(args.data_path)
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        df = pd.DataFrame({
            'id': range(config['data']['n_rows']),
            'value1': np.random.normal(100, 15, config['data']['n_rows']),
            'value2': np.random.exponential(2, config['data']['n_rows']),
            'category': np.random.choice(['A', 'B', 'C'], config['data']['n_rows'])
        })
    else:
        raise ValueError("No data source specified")
    
        logging.info(f"Shape: {df.shape}")
    logging.info(f"\nColumn Types:")
    logging.info(df.dtypes)
    logging.info(f"\nSummary Statistics:")
    logging.info(df.describe())
    
    stats = calculate_summary_statistics(df)
    logging.info(f"\nMemory Usage: {stats['memory_usage'] / 1024**2:.2f} MB")
    
        for col in df.select_dtypes(include=[np.number]).columns:
            plot_data_distribution(df, col, f"Distribution of {col}",
                                 output_dir / f'distribution_{col}.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

