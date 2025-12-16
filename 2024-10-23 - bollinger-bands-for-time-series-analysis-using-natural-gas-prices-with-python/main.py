#!/usr/bin/env python3
"""
Bollinger Bands for Time Series Analysis

Main entry point for running Bollinger Bands analysis.
"""

import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_synthetic_prices,
    calculate_bollinger_bands,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Bollinger Bands Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path, parse_dates=True, index_col=0)
        target_col = config['data']['target_column']
    else:
                df = generate_synthetic_prices(
            config['data']['start_date'],
            config['data']['end_date'],
            config['data']['frequency'],
            config['data']['initial_price'],
            config['data']['volatility'],
            config['data']['seed']
        )
        target_col = config['data']['target_column']
    
    df = calculate_bollinger_bands(
        df,
        config['bollinger_bands']['window'],
        config['bollinger_bands']['num_std'],
        target_col
    )
    
    plot_bollinger_bands(df, target_col, config['bollinger_bands']['window'],
                        output_dir / 'bollinger_bands.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

