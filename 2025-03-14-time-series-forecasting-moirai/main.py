#!/usr/bin/env python3
"""
Moirai Time Series Forecasting

Main entry point for running Moirai forecasting analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    load_data,
    create_dataset,
    split_dataset,
    create_moirai_model,
    generate_forecasts,
    save_forecast_plot
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Moirai Time Series Forecasting')
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
    
        df = load_data(data_path, config['data']['date_column'], 
                  config['data']['value_column'], config['data']['frequency'])
    
        dataset = create_dataset(df, config['data']['value_column'])
    
        train, test_data = split_dataset(dataset, config['test']['test_size'])
    
        model = create_moirai_model(
        dataset,
        config['model']['size'],
        config['model']['prediction_length'],
        config['model']['context_length'],
        config['model']['patch_size'],
        config['model']['num_samples']
    )
    
        forecasts = generate_forecasts(model, test_data, config['model']['batch_size'])
    
        save_forecast_plot(test_data, forecasts, config['model']['context_length'],
                      output_dir / 'moirai_forecast.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

