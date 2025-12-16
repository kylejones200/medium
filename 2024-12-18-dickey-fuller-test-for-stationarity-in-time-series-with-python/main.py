#!/usr/bin/env python3
"""
Dickey-Fuller Test for Stationarity in Time Series

Main entry point for running stationarity tests.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_random_walk,
    calculate_rolling_stats,
    test_stationarity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Dickey-Fuller Test for Stationarity')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    df = generate_random_walk(config['data']['n_samples'], config['data']['seed'])
    
    plot_time_series(df, output_dir / 'simulated_timeseries.png')
    
    stats_df = calculate_rolling_stats(df, config['data']['rolling_window'])
    plot_rolling_stats(stats_df, output_dir / 'rolling_stats.png')
    
    result = test_stationarity(df['value'])
    
            logging.info("Is Stationary:", result['is_stationary'])
    for key, value in result['critical_values'].items():
        logging.info(f'Critical Value ({key}): {value:.4f}')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

