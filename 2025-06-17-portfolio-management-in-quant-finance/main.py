#!/usr/bin/env python3
"""
Portfolio Management in Quantitative Finance

Main entry point for running portfolio analysis.
"""

import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_asset_returns,
    efficient_frontier,
    calculate_alpha_beta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Portfolio Management")
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
        returns, mean_returns, cov_matrix = generate_asset_returns(
        config['simulation']['asset_returns']['n_assets'],
        config['simulation']['asset_returns']['n_periods'],
        config['simulation']['asset_returns']['seed']
    )
    
        results, _ = efficient_frontier(
        mean_returns,
        cov_matrix,
        config['simulation']['efficient_frontier']['n_portfolios'],
        config['simulation']['efficient_frontier']['seed']
    )
    plot_efficient_frontier(results, output_dir / 'efficient_frontier.png')
    
        if config['simulation']['alpha_beta']['seed'] is not None:
        np.random.seed(config['simulation']['alpha_beta']['seed'])
    market_returns = np.random.normal(
        config['simulation']['alpha_beta']['market_mean'],
        config['simulation']['alpha_beta']['market_std'],
        size=config['simulation']['alpha_beta']['n_periods']
    )
    asset_returns = (config['simulation']['alpha_beta']['asset_beta'] * market_returns + 
                    np.random.normal(0, config['simulation']['alpha_beta']['asset_noise_std'], 
                                   size=config['simulation']['alpha_beta']['n_periods']))
    alpha, beta = calculate_alpha_beta(
        asset_returns,
        market_returns,
        config['simulation']['alpha_beta']['risk_free']
    )
    logging.info(f"Alpha: {alpha:.4f}, Beta: {beta:.2f}")
    plot_alpha_beta(market_returns, asset_returns, alpha, beta,
                   output_dir / 'alpha_beta.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

