#!/usr/bin/env python3
"""
Transfer Entropy with Tech ETFs

Main entry point for running VECM and cointegration analysis.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    download_etf_data,
    perform_adf_test,
    perform_johansen_test,
    fit_vecm_model,
    plot_etf_prices,
    plot_irf
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Transfer Entropy with Tech ETFs')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        print(f"Loading data from {args.data_path}...")
        data = pd.read_csv(args.data_path, parse_dates=["Date"], index_col="Date")
    else:
        print("Downloading ETF data...")
        data = download_etf_data(
            config['data']['tickers'],
            config['data']['start_date'],
            config['data']['end_date']
        )
        data.columns = ['Tech_ETF', 'Semiconductor_ETF']
        data.to_csv(config['data']['output_file'])
    
    plot_etf_prices(data, output_dir / 'xlk_smh_prices.png')
    
    if config['analysis']['adf_test']:
        print("\nPerforming ADF tests...")
        for col in data.columns:
            perform_adf_test(data[col], col)
    
    if config['analysis']['johansen_test']:
        print("\nPerforming Johansen cointegration test...")
        perform_johansen_test(data)
    
    if config['analysis']['vecm']['enabled']:
        print("\nFitting VECM model...")
        vec_res = fit_vecm_model(
            data,
            config['analysis']['vecm']['k_ar_diff'],
            config['analysis']['vecm']['coint_rank'],
            config['analysis']['vecm']['deterministic']
        )
        print(vec_res.summary())
        
        if config['analysis']['irf']['enabled']:
            print("\nComputing Impulse Response Functions...")
            irf = vec_res.irf(config['analysis']['irf']['periods'])
            plot_irf(irf, output_dir / 'vecm_irf.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    import pandas as pd
    main()

