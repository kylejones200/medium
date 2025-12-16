#!/usr/bin/env python3
"""
Time Series Visualization for Business Analysis

Main entry point for running time series visualizations.
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
    parser = argparse.ArgumentParser(description='Time Series Visualization for Business Analysis')
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
        values = np.sin(np.arange(config['data']['n_periods']) / 30) * 100 + 500 + np.random.normal(0, 20, config['data']['n_periods'])
        df = pd.DataFrame({'date': dates, 'value': values})
    else:
        raise ValueError("No data source specified")
    
    if config['visualization']['time_series']:
                plot_time_series(df, 'value', 'date', "Business Time Series",
                        output_dir / 'time_series.png')
    
    if config['visualization']['seasonal_decomposition']:
                plot_seasonal_decomposition(df, 'value', config['visualization']['seasonal_period'],
                                   "Seasonal Decomposition", output_dir / 'seasonal_decomposition.png')
    
    logging.info(f"\nVisualization complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

