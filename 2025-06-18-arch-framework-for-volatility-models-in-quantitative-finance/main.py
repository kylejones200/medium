#!/usr/bin/env python3
"""
ARCH Framework for Volatility Models in Quantitative Finance

Main entry point for running ARCH/GARCH volatility modeling.
"""

import argparse
import yaml
import logging
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
    parser = argparse.ArgumentParser(description='ARCH Framework for Volatility Models')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        returns = pd.read_csv(args.data_path, index_col=0, parse_dates=True).iloc[:, 0]
    elif config['data']['generate_synthetic']:
                returns, volatility = simulate_returns_volatility(config['data']['n_periods'],
                                                         config['data']['seed'])
    else:
        raise ValueError("No data source specified")
    
        garch_fit = fit_garch_model(returns, config['model']['p'], config['model']['q'])
    logging.info(garch_fit.summary())
    
        forecast_var = forecast_volatility(garch_fit, config['forecast']['horizon'])
    
    if config['data']['generate_synthetic']:
        plot_volatility_analysis(returns, volatility, forecast_var,
                               "ARCH Framework: Volatility Modeling",
                               output_dir / 'volatility_analysis.png')
    else:
        volatility = pd.Series(np.abs(returns), index=returns.index)
        plot_volatility_analysis(returns, volatility, forecast_var,
                               "ARCH Framework: Volatility Modeling",
                               output_dir / 'volatility_analysis.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

