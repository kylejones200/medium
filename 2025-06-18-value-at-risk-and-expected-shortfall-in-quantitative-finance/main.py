#!/usr/bin/env python3
"""
Value at Risk and Expected Shortfall in Quantitative Finance

Main entry point for running VaR and ES analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    calculate_var_parametric,
    calculate_var_historical,
    calculate_var_monte_carlo,
    calculate_expected_shortfall,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Value at Risk and Expected Shortfall')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to returns data')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        logging.info(f"Loading data from {args.data_path}...")
        returns = pd.read_csv(args.data_path)['returns']
    elif config['data']['generate_synthetic']:
                np.random.seed(config['data']['seed'])
        returns = pd.Series(np.random.normal(0.001, 0.02, config['data']['n_periods']))
    else:
        raise ValueError("No data source specified")
    
    var_results = {}
    
    if 'parametric' in config['var']['methods']:
                var_results['parametric'] = calculate_var_parametric(
            returns, config['var']['confidence_level'], config['var']['time_horizon_days']
        )
        logging.info(f"Parametric VaR: {var_results['parametric']['var_pct']:.4f}%")
    
    if 'historical' in config['var']['methods']:
                var_results['historical'] = calculate_var_historical(returns, config['var']['confidence_level'])
        logging.info(f"Historical VaR: {var_results['historical']['var_pct']:.4f}%")
    
    if 'monte_carlo' in config['var']['methods']:
                var_results['monte_carlo'] = calculate_var_monte_carlo(
            returns, config['var']['confidence_level'],
            config['var']['monte_carlo']['n_simulations'], config['data']['seed']
        )
        logging.info(f"Monte Carlo VaR: {var_results['monte_carlo']['var_pct']:.4f}%")
    
        es = calculate_expected_shortfall(returns, config['expected_shortfall']['confidence_level'])
    logging.info(f"Expected Shortfall: {es:.4f}%")
    
    if len(var_results) > 1:
        plot_var_comparison(var_results, output_dir / 'var_comparison.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

