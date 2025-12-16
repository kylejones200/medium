#!/usr/bin/env python3
"""
Digital Integration for Oil Production Monitoring

Main entry point for running oil production monitoring analysis.
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
    parser = argparse.ArgumentParser(description='Digital Integration for Oil Production Monitoring')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df = analyze_production_data(df, config['data']['timestamp_column'],
                                    config['data']['production_column'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='H')
        base_production = 100 + 20 * np.sin(np.arange(config['data']['n_periods']) / 24)
        production = base_production + np.random.normal(0, 5, config['data']['n_periods'])
        production = np.maximum(production, 0)
        
        df = pd.DataFrame({
            config['data']['timestamp_column']: dates,
            config['data']['production_column']: production
        })
        df = analyze_production_data(df, config['data']['timestamp_column'],
                                    config['data']['production_column'])
    else:
        raise ValueError("No data source specified")
    
        metrics = calculate_production_metrics(df, config['data']['production_column'])
    
    logging.info(f"\nProduction Metrics:")
    logging.info(f"Total Production: {metrics['total_production']:.2f}")
    logging.info(f"Mean Production: {metrics['mean_production']:.2f}")
    logging.info(f"Volatility: {metrics['volatility']:.4f}")
    logging.info(f"Trend: {metrics['trend']}")
    
    if config['monitoring']['check_trend']:
        current_prod = df[config['data']['production_column']].iloc[-1]
        max_prod = metrics['max_production']
        if current_prod < config['monitoring']['alert_threshold'] * max_prod:
            logging.info(f"\n⚠️  Alert: Production below threshold ({current_prod:.2f} < {config['monitoring']['alert_threshold'] * max_prod:.2f})")
    
    plot_production_monitoring(df, config['data']['production_column'],
                             "Oil Production Monitoring", output_dir / 'production_monitoring.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

