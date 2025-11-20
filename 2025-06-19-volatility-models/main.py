#!/usr/bin/env python3
"""
ARCH Framework for Volatility Models

Main entry point for running ARCH volatility modeling.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.core import (
    simulate_returns_with_volatility_clustering,
    fit_arch_model,
    forecast_volatility,
    plot_returns_volatility,
    plot_volatility_forecast
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='ARCH Volatility Models')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Simulating returns with volatility clustering...")
    returns, volatility = simulate_returns_with_volatility_clustering(
        config['simulation']['n'],
        config['simulation']['omega'],
        config['simulation']['alpha'],
        config['simulation']['seed']
    )
    
    data = pd.DataFrame({"returns": returns, "volatility": volatility})
    plot_returns_volatility(returns, volatility, output_dir / 'simulated_returns_volatility.png')
    
    print("Fitting ARCH model...")
    arch_model_fit = fit_arch_model(data["returns"], config['model']['vol_type'], config['model']['p'])
    print(arch_model_fit.summary())
    
    print("Forecasting volatility...")
    forecast_variance = forecast_volatility(arch_model_fit, config['forecast']['horizon'])
    plot_volatility_forecast(forecast_variance, output_dir / 'forecasted_volatility.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

