#!/usr/bin/env python3
"""
Gas Prices vs Unemployment Driving Analysis

Main entry point for running economic relationship analysis.
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
    parser = argparse.ArgumentParser(description='Gas Prices vs Unemployment Driving Analysis')
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
        gas_prices = np.random.normal(3.5, 0.5, config['data']['n_periods'])
        unemployment = 5.0 + 0.5 * gas_prices + np.random.normal(0, 0.3, config['data']['n_periods'])
        
        df = pd.DataFrame({
            config['data']['gas_column']: gas_prices,
            config['data']['unemployment_column']: unemployment
        })
    else:
        raise ValueError("No data source specified")
    
    logging.info("Preparing economic data...")
    gas, unemployment = prepare_economic_data(df, config['data']['gas_column'],
                                            config['data']['unemployment_column'])
    
    logging.info("Analyzing correlation...")
    results = analyze_correlation(gas, unemployment)
    
    logging.info("Economic Analysis Results:")
    logging.info(f"Correlation: {results['correlation']:.4f}")
    logging.info(f"R² Score: {results['r2']:.4f}")
    logging.info(f"Slope: {results['slope']:.4f}")
    
    if abs(results['correlation']) > config['analysis']['correlation_threshold']:
        direction = "positive" if results['correlation'] > 0 else "negative"
        logging.info(f"✓ Significant {direction} correlation detected")
    
    plot_economic_relationship(gas, unemployment, "Gas Prices vs Unemployment Rate",
                              output_dir / 'economic_relationship.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

