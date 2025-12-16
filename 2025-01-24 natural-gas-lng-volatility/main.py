#!/usr/bin/env python3
"""
Natural Gas and LNG Volatility Analysis

Main entry point for running natural gas and LNG volatility analysis.
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
    parser = argparse.ArgumentParser(description='Natural Gas and LNG Volatility Analysis')
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
        base_price = 3.0
        natural_gas = base_price + np.cumsum(np.random.normal(0, 0.1, config['data']['n_periods']))
        lng = natural_gas * 1.2 + np.random.normal(0, 0.2, config['data']['n_periods'])
        
        df = pd.DataFrame({
            'date': dates,
            config['data']['price_columns'][0]: natural_gas,
            config['data']['price_columns'][1]: lng
        })
    else:
        raise ValueError("No data source specified")
    
        results = analyze_price_relationships(df, config['data']['price_columns'])
    
    logging.info(f"\nVolatilities:")
    for col, vol in results['volatilities'].items():
        logging.info(f"  {col}: {vol:.4f}")
    
    logging.info(f"\nCorrelations:")
    logging.info(results['correlations'])
    
    corr_value = results['correlations'].iloc[0, 1]
    if abs(corr_value) > config['analysis']['correlation_threshold']:
        logging.info(f"\n✓ Strong correlation detected: {corr_value:.4f}")
    
    plot_volatility_analysis(df, config['data']['price_columns'],
                           "Natural Gas and LNG Volatility Analysis",
                           output_dir / 'volatility_analysis.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

