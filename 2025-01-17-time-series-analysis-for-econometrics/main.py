#!/usr/bin/env python3
"""
Time Series Analysis for Econometrics

Main entry point for running econometric time series analysis.
"""

import argparse
import yaml
import logging
import numpy as np
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
    parser = argparse.ArgumentParser(description='Time Series Analysis for Econometrics')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2020-01-01', periods=config['data']['n_periods'], freq='D')
        x = np.cumsum(np.random.normal(0, 1, config['data']['n_periods']))
        y = 0.5 * x + np.random.normal(0, 0.5, config['data']['n_periods'])
        data = pd.DataFrame({'x': x, 'y': y}, index=dates)
    else:
        raise ValueError("No data source specified")
    
    plot_time_series(data, "Time Series Data", output_dir / 'time_series.png')
    
    if config['analysis']['adf_test']:
                for col in data.columns:
            result = perform_adf_test(data[col])
            logging.info(f"{col}: ADF={result['adf_statistic']:.4f}, p-value={result['p_value']:.4f}, stationary={result['is_stationary']}")
    
    if config['analysis']['cointegration_test']:
                result = perform_cointegration_test(data['x'], data['y'])
        logging.info(f"Cointegration: test_stat={result['test_statistic']:.4f}, p-value={result['p_value']:.4f}, cointegrated={result['is_cointegrated']}")
    
    if config['analysis']['var_model']['enabled']:
                var_result = fit_var_model(data, config['analysis']['var_model']['maxlags'])
        logging.info(f"\n{var_result.summary()}")
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

