#!/usr/bin/env python3
"""
Modernizing Grid Intelligence with NERC CIP

Main entry point for running grid intelligence analysis.
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
    parser = argparse.ArgumentParser(description='Modernizing Grid Intelligence with NERC CIP')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df = analyze_grid_data(df, config['data']['timestamp_column'], config['data']['value_column'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='H')
        values = 1000 + 100 * np.sin(np.arange(config['data']['n_periods']) / 24) + np.random.normal(0, 20, config['data']['n_periods'])
        df = pd.DataFrame({
            config['data']['timestamp_column']: dates,
            config['data']['value_column']: values
        })
        df = analyze_grid_data(df, config['data']['timestamp_column'], config['data']['value_column'])
    else:
        raise ValueError("No data source specified")
    
        metrics = calculate_grid_metrics(df, config['data']['value_column'])
    logging.info(f"\nGrid Metrics:")
    logging.info(f"Mean: {metrics['mean']:.2f}")
    logging.info(f"Volatility: {metrics['volatility']:.4f}")
    
    if config['nerc_cip']['compliance_check']:
                    plot_grid_intelligence(df, config['data']['value_column'], "Grid Intelligence Data",
                          output_dir / 'grid_intelligence.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

