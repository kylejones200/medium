#!/usr/bin/env python3
"""
Modeling Heating Oil and Crude Oil Price Relationships Using VECM

Main entry point for running VECM analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    fetch_data,
    test_stationarity,
    test_cointegration,
    fit_vecm_model,
    forecast_vecm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='VECM Modeling: Heating Oil vs Crude Oil')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Fetching data from Yahoo Finance...")
    df = fetch_data(config['data']['symbols'], 
                   config['data']['start_date'], 
                   config['data']['end_date'])
    
    logging.info(f"Data shape: {df.shape}")
    logging.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    logging.info("Testing stationarity...")
    stationarity = test_stationarity(df)
    for col, results in stationarity.items():
        logging.info(f"{col}:")
        logging.info(f"  Level - ADF: {results['level']['statistic']:.4f}, p={results['level']['pvalue']:.4f}")
        logging.info(f"  Diff - ADF: {results['diff']['statistic']:.4f}, p={results['diff']['pvalue']:.4f}")
    
    logging.info("Testing cointegration...")
    coint_result = test_cointegration(df)
    logging.info(f"Trace Statistic: {coint_result['trace_statistic']}")
    logging.info(f"Critical Values (90%, 95%, 99%): {coint_result['critical_values']}")
    
    logging.info("Fitting VECM model...")
    vecm_model = fit_vecm_model(df, config['model']['lags'], config['model']['coint_rank'])
    logging.info(f"\n{vecm_model.summary()}")
    
    logging.info(f"Forecasting {config['forecast']['steps']} steps ahead...")
    forecast = forecast_vecm(vecm_model, df, config['forecast']['steps'])
    
    plot_price_relationship(df, forecast, "Heating Oil vs Crude Oil: VECM Analysis",
                           output_dir / 'vecm_analysis.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

