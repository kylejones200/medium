#!/usr/bin/env python3
"""
ARAR Algorithm for H-Step Forecasting

Main entry point for running ARAR forecasting analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    load_data,
    apply_differencing,
    select_reduced_lags,
    fit_arar_model,
    generate_arar_forecast,
    fit_arima_model,
    generate_arima_forecast,
    calculate_mape,
)
from statsmodels.tsa.stattools import acf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='ARAR Algorithm for H-Step Forecasting')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_path = args.data_path if args.data_path else Path(config['data']['source'])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
        y = load_data(data_path, config['data']['date_column'], config['data']['value_column'])
    
    if config['data']['resample_freq']:
        y = y.resample(config['data']['resample_freq']).mean()
        y = y.asfreq(config['data']['resample_freq'])
    
    h = config['forecast']['horizon']
    test_size = config['forecast']['test_size'] or h
    
    train = y.iloc[:-test_size]
    test = y.iloc[-test_size:] if test_size > 0 else None
    
                z = apply_differencing(y)
        plot_series_comparison(y, z, output_dir / 'arar_series_visualization.png')
    
    if config['analysis']['run_arar']:
                z_train = apply_differencing(train)
        
        acf_vals = acf(z_train, nlags=config['model']['arar']['max_lag'])
        
        if config['model']['arar']['lag_selection_strategy'] == "powers_of_2":
            lags = config['model']['arar']['lags']
        else:
            lags = select_reduced_lags(acf_vals, config['model']['arar']['max_lag'],
                                     config['model']['arar']['lag_selection_strategy'])
        
        arar_model = fit_arar_model(z_train, lags)
        logging.info(arar_model.summary())
        
        y_forecast_arar = generate_arar_forecast(arar_model, h, train.iloc[-1])
        forecast_index = pd.date_range(start=train.index[-1], periods=h+1, 
                                      freq=train.index.freq or 'D')[1:]
        y_forecast_arar_series = pd.Series(y_forecast_arar, index=forecast_index)
        
        plot_forecast_comparison(y, y_forecast_arar, forecast_index, h,
                               output_dir / 'arar_forecast_plot.png')
        
        if test is not None and len(test) == h:
            mape_arar = calculate_mape(test, y_forecast_arar_series)
            logging.info(f"ARAR MAPE: {mape_arar:.4f}")
        
        if config['analysis']['run_arima_comparison']:
                        arima_model = fit_arima_model(train, tuple(config['model']['arima']['order']))
            y_forecast_arima = generate_arima_forecast(arima_model, h)
            y_forecast_arima.index = forecast_index
            
            if test is not None and len(test) == h:
                mape_arima = calculate_mape(test, y_forecast_arima)
                logging.info(f"ARIMA MAPE: {mape_arima:.4f}")
                
                plot_arar_vs_arima(y, train, test, y_forecast_arar_series,
                                 y_forecast_arima, forecast_index,
                                 mape_arar, mape_arima,
                                 output_dir / 'arar_vs_arima_forecast.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

