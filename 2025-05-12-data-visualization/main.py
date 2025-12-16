#!/usr/bin/env python3
"""
Data Visualization

Main entry point for running data visualization examples.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import (
    generate_sales_data,
)
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Data Visualization")
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        logging.info(f"Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
    else:
                df = generate_sales_data(
            config['data']['months'],
            config['data']['regions'],
            config['data']['seed']
        )
    
    if config['visualization']['seaborn_line']:
                plot_seaborn_line(df, 'Month', 'Sales', 'Region', output_dir / 'sales_seaborn.png')
    
    if config['visualization']['weekly_trend'] or config['visualization']['store_comparison']:
        raw = pd.DataFrame({
            'Store': ['North', 'South', 'East', 'West'],
            'Week_1': [300, 250, 400, 375],
            'Week_2': [310, 245, 390, 380],
            'Week_3': [305, 260, 395, 370]
        })
        
        long = pd.melt(raw, id_vars='Store', var_name='Week', value_name='Sales')
        long['Week'] = long['Week'].str.replace('Week_', '').astype(int)
        
        if config['visualization']['weekly_trend']:
                        plot_weekly_trend(long, output_dir / 'weekly_sales.png')
        
        if config['visualization']['store_comparison']:
                        plot_store_comparison(long, output_dir / 'store_weekly_sales.png')
    
    logging.info(f"\nVisualization complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

