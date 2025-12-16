#!/usr/bin/env python3
"""
Forecast Error Variance Analysis for Time Series Using UK Births Data

Main entry point for running forecast error variance analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    load_and_prepare_data,
    fit_forecast_births,
    calculate_error_metrics,
    calculate_multi_step_variance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Forecast Error Variance Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path:
        births = load_and_prepare_data(file_path=args.data_path)
    elif config['data']['url']:
        births = load_and_prepare_data(url=config['data']['url'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        births = load_and_prepare_data()
    else:
        raise ValueError("No data source specified")
    
    logging.info("Fitting exponential smoothing model...")
    fitted_births = fit_forecast_births(births['Births'], config['model']['seasonal_periods'])
    forecast_errors = births['Births'] - fitted_births
    
    logging.info("Calculating error metrics...")
    metrics = calculate_error_metrics(forecast_errors, config['model']['alpha'])
    
    logging.error(" Metrics:")
    logging.info(f"Mean Error: {metrics['mean_error']:.4f}")
    logging.info(f"MAD: {metrics['mad']:.4f}")
    logging.info(f"Sample Variance: {metrics['sample_variance']:.4f}")
    
    var_3 = calculate_multi_step_variance(config['analysis']['c1'], 
                                         config['analysis']['c_tau_3'],
                                         metrics['mad'], config['model']['alpha'])
    var_6 = calculate_multi_step_variance(config['analysis']['c1'],
                                         config['analysis']['c_tau_6'],
                                         metrics['mad'], config['model']['alpha'])
    var_12 = calculate_multi_step_variance(config['analysis']['c1'],
                                          config['analysis']['c_tau_12'],
                                          metrics['mad'], config['model']['alpha'])
    
    logging.info("Multi-step Forecast Error Variance:")
    logging.info(f"3-step: {var_3:.4f}")
    logging.info(f"6-step: {var_6:.4f}")
    logging.info(f"12-step: {var_12:.4f}")
    
    plot_forecast_analysis(births['Births'], fitted_births, forecast_errors,
                          "Forecast Error Variance Analysis: UK Births",
                          output_dir / 'forecast_analysis.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

