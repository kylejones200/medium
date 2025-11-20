#!/usr/bin/env python3
"""
Exploring Volatility with ARCH Models for Time Series Forecasting

Main entry point for running ARCH/GARCH volatility modeling.

Usage:
    python main.py
    python main.py --data-path data/returns.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import simulate_volatility_clustering, fit_garch_model, forecast_volatility, plot_volatility_analysis


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Exploring Volatility with ARCH Models')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        returns = pd.read_csv(args.data_path)['returns']
    elif config['data']['generate_synthetic']:
        print("Simulating returns with volatility clustering...")
        returns, volatility = simulate_volatility_clustering(
            config['data']['n_periods'],
            config['model']['omega'],
            config['model']['alpha'],
            config['model']['beta'],
            config['data']['seed']
        )
        returns = pd.Series(returns)
    else:
        raise ValueError("No data source specified")
    
    print("Fitting GARCH model...")
    garch_fit = fit_garch_model(returns, config['model']['p'], config['model']['q'])
    print(garch_fit.summary())
    
    print("Forecasting volatility...")
    forecast_variance = forecast_volatility(garch_fit, config['forecast']['horizon'])
    
    if config['data']['generate_synthetic']:
        plot_volatility_analysis(returns.values, volatility, forecast_variance,
                               output_dir / 'volatility_analysis.png')
    else:
        from src.core import plot_volatility_analysis
        plot_volatility_analysis(returns.values, returns.values, forecast_variance,
                               output_dir / 'volatility_analysis.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

