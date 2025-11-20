#!/usr/bin/env python3
"""
VAR, IRF, and FEVD Analysis for Gold, Newmont, and GDX

Main entry point for running Vector Autoregression analysis with impulse response
functions and forecast error variance decomposition.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    fetch_yfinance_data,
    compute_log_returns,
    test_stationarity,
    fit_var_model,
    plot_irf,
    plot_fevd,
    plot_cumulative_irf
)
from statsmodels.tsa.stattools import grangercausalitytests


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='VAR, IRF, and FEVD Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Fetching data from Yahoo Finance...")
    df = fetch_yfinance_data(
        config['data']['tickers'],
        config['data']['start_date'],
        config['data']['end_date']
    )
    df.columns = config['data']['column_names']
    
    print("Computing log returns...")
    log_returns = compute_log_returns(df, config['data']['resample_freq'])
    
    print("\nStationarity Tests:")
    for col in log_returns.columns:
        result = test_stationarity(log_returns[col], col)
        print(f"{col}: p-value = {result['p_value']:.4f}, "
             f"Stationary: {result['is_stationary']}")
    
    if config['analysis']['granger_causality']['enabled']:
        print("\nGranger Causality Tests:")
        for test in config['analysis']['granger_causality']['test_pairs']:
            print(f"\n{test['description']}")
            try:
                test_data = log_returns[[test['y'], test['x']]].dropna()
                grangercausalitytests(test_data, 
                                     maxlag=config['analysis']['granger_causality']['maxlag'],
                                     verbose=True)
            except Exception as e:
                print(f"Error: {e}")
    
    print("\nFitting VAR model...")
    model, fitted_model = fit_var_model(log_returns, config['model']['max_lag'])
    
    print(f"\nVAR Model Summary (Selected Lag: {fitted_model.k_ar}):")
    print(fitted_model.summary())
    
    print("\nResidual Correlation Matrix:")
    print(fitted_model.resid.corr())
    
    if config['output']['plot_irf']:
        print("\nComputing Impulse Response Functions...")
        irf = fitted_model.irf(config['analysis']['irf']['periods'])
        plot_irf(irf, output_dir / 'irf_plot.png')
    
    if config['output']['plot_fevd']:
        print("Computing Forecast Error Variance Decomposition...")
        fevd = fitted_model.fevd(config['analysis']['fevd']['periods'])
        plot_fevd(fevd, output_dir / 'fevd_plot.png')
    
    if config['output']['plot_cumulative_irf']:
        print("Plotting cumulative IRFs...")
        plot_cumulative_irf(
            irf,
            fitted_model,
            config['output']['cumulative_irf_shock'],
            config['output']['cumulative_irf_responses'],
            config['analysis']['irf']['periods'],
            output_dir / 'cumulative_irf_gold_shocks.png'
        )
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

