#!/usr/bin/env python3
"""
Time Series Analysis with statsmodels in Python

Main entry point for running time series analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import (
    load_data,
    split_data,
    test_stationarity,
    decompose_series,
    fit_arima,
    forecast_arima,
    fit_holt_winters,
    forecast_holt_winters,
    calculate_mape,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Time Series Analysis with statsmodels')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_path = args.data_path if args.data_path else (
        Path(config['data']['source']) if config['data']['source'] else None
    )
    
    df = load_data(data_path)
    
    train, hold_out = split_data(df, config['model']['hold_out_days'])
    
    plot_time_series(df, train, hold_out, output_dir / 'simulated_time_series.png')
    
    if config['analysis']['run_stationarity_test']:
        stationarity = test_stationarity(df["value"])
        logging.info(f"ADF Statistic: {stationarity['adf_statistic']:.4f}")
        logging.info(f"P-Value: {stationarity['p_value']:.4f}")
        if stationarity['is_stationary']:
            logging.info("The time series is stationary.")
        else:
            logging.info("The time series is non-stationary.")
    
    if config['analysis']['run_decomposition']:
        decomposition = decompose_series(
            df["value"],
            period=config['model']['decomposition_period']
        )
        plot_decomposition(decomposition, output_dir / 'time_series_decomposition.png')
    
    if config['analysis']['run_acf_pacf']:
        plot_acf_pacf(df["value"], lags=30, output_path=output_dir / 'acf_pacf_plots.png')
    
    if config['analysis']['run_arima']:
        arima_model = fit_arima(
            train["value"],
            order=tuple(config['model']['arima_order']),
            freq=config['data']['frequency']
        )
        
        forecast = forecast_arima(arima_model, config['model']['hold_out_days'], hold_out.index)
        mape = calculate_mape(hold_out["value"], forecast['mean'])
        logging.info(f"ARIMA MAPE: {mape:.3%}")
        
        plot_arima_forecast(train, hold_out, forecast, mape, output_dir / 'arima_forecast_holdout.png')
        
        if config['analysis']['run_residuals_diagnostics']:
            plot_residuals_diagnostics(arima_model, output_dir / 'arima_residuals_diagnostics.png')
    
    if config['analysis']['run_holt_winters']:
        hw_model = fit_holt_winters(train["value"], config['model']['seasonal_periods'])
        hw_forecast = forecast_holt_winters(hw_model, config['model']['hold_out_days'])
        mape_hw = calculate_mape(hold_out["value"], hw_forecast)
        logging.info(f"Holt-Winters MAPE: {mape_hw:.3%}")
        
        plot_holt_winters_forecast(train, hold_out, hw_forecast, mape_hw, output_dir / 'holt_winters_forecast.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

