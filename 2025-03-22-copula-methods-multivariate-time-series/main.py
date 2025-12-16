#!/usr/bin/env python3
"""
Copula Methods for Multivariate Time Series

Main entry point for running copula analysis.
"""

import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from src.core import (
    generate_stock_interest_data,
    generate_inflation_unemployment_data,
    transform_to_uniform,
    fit_clayton_copula,
    fit_studentt_copula,
    simulate_copula_forecast,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Copula Methods for Multivariate Time Series')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Generating stock and interest rate data...")
    stock_data = generate_stock_interest_data(
        config['data']['time_steps'],
        config['data']['seed']
    )
    
    u = transform_to_uniform(stock_data['Stock Returns'], config['data']['time_steps'])
    v = transform_to_uniform(stock_data['Interest Rates'], config['data']['time_steps'])
    
    logging.info("Fitting Clayton copula...")
    copula = fit_clayton_copula(u, v)
    
    u_future = np.random.uniform(size=config['copula']['stock_interest']['n_samples'])
    returns_forecast, rates_forecast = simulate_copula_forecast(
        copula, u_future, stock_data, 'Stock Returns', 'Interest Rates',
        config['copula']['stock_interest']['n_samples']
    )
    
    plot_copula_forecast(
        returns_forecast, rates_forecast,
        "Forecasted Stock Returns", "Forecasted Interest Rates",
        "Stock Returns vs. Interest Rates (Copula Forecast)",
        output_dir / 'copula_forecast_stock_interest.png'
    )
    
    logging.info("Generating inflation and unemployment data...")
    inflation_data = generate_inflation_unemployment_data(
        config['data']['time_steps'],
        config['data']['seed']
    )
    
    u = transform_to_uniform(inflation_data['Inflation'], config['data']['time_steps'])
    v = transform_to_uniform(inflation_data['Unemployment'], config['data']['time_steps'])
    
    logging.info("Fitting Student-t copula...")
    copula = fit_studentt_copula(u, v)
    
    u_future = np.random.uniform(size=config['copula']['inflation_unemployment']['n_samples'])
    inflation_forecast, unemployment_forecast = simulate_copula_forecast(
        copula, u_future, inflation_data, 'Inflation', 'Unemployment',
        config['copula']['inflation_unemployment']['n_samples']
    )
    
    plot_copula_forecast(
        inflation_forecast, unemployment_forecast,
        "Forecasted Inflation", "Forecasted Unemployment",
        "Inflation vs. Unemployment (t-Copula Forecast)",
        output_dir / 'copula_forecast_inflation_unemployment.png'
    )
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

