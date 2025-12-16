#!/usr/bin/env python3
"""
Tidy Data: Long and Wide Format Transformations

Main entry point for demonstrating data reshaping operations.
"""

import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from src.core import (
    generate_wide_data,
    wide_to_long,
    long_to_wide,
    pivot_table_aggregation,
    groupby_aggregation,
    generate_weekly_sales_data,
    reshape_weekly_data,
    plot_weekly_trend,
    plot_store_comparison
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Tidy Data Transformations')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if config['transformations']['wide_to_long']:
        wide_df = generate_wide_data(config['data']['stores'], config['data']['months'])
        long_df = wide_to_long(wide_df, 'Store', 'Month', 'Sales', '_Sales')
        logging.info(f"Wide format:\n{wide_df}")
        logging.info(f"Long format:\n{long_df}")
    
    if config['transformations']['long_to_wide']:
        wide_df = long_to_wide(long_df, 'Store', 'Month', 'Sales')
        logging.info(f"Wide format:\n{wide_df}")
    
    if config['transformations']['pivot_table']:
        pivot_df = pivot_table_aggregation(long_df, 'Store', 'Month', 'Sales', 'sum')
        logging.info(f"Pivot table:\n{pivot_df}")
    
    if config['transformations']['groupby']:
        data = pd.DataFrame({
            'Store': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Month': ['Jan', 'Feb', 'Mar', 'Jan', 'Feb', 'Mar'],
            'Sales': [100, 120, 130, 90, 100, 110]
        })
        store_agg = groupby_aggregation(data, ['Store'], 'Sales', 
                                       {'avg': 'mean', 'total': 'sum', 'volatility': 'std'})
        logging.info(f"Store aggregation:\n{store_agg}")
    
    if config['transformations']['weekly_analysis']:
        raw = generate_weekly_sales_data(config['data']['weekly_stores'], config['data']['weeks'])
        long = reshape_weekly_data(raw)
        
        plot_weekly_trend(long, output_dir / 'weekly_sales.png')
        plot_store_comparison(long, output_dir / 'store_weekly_sales.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()
