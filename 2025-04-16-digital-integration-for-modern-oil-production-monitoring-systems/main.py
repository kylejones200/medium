#!/usr/bin/env python3
"""
Digital Integration for Modern Oil Production Monitoring Systems

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
    parser = argparse.ArgumentParser(description='Digital Integration Oil Production Monitoring')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df = analyze_production_monitoring(df, config['data']['timestamp_column'],
                                          config['data']['production_column'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='H')
        base_production = 1000 + 100 * np.sin(np.arange(config['data']['n_periods']) / 24)
        production = base_production + np.random.normal(0, 50, config['data']['n_periods'])
        production = np.maximum(production, 0)
        
        df = pd.DataFrame({
            config['data']['timestamp_column']: dates,
            config['data']['production_column']: production
        })
        df = analyze_production_monitoring(df, config['data']['timestamp_column'],
                                          config['data']['production_column'])
    else:
        raise ValueError("No data source specified")
    
        kpis = calculate_production_kpis(df, config['data']['production_column'])
    
    logging.info(f"\nProduction KPIs:")
    logging.info(f"Total Production: {kpis['total_production']:.2f}")
    logging.info(f"Mean Production: {kpis['mean_production']:.2f}")
    logging.info(f"Peak Production: {kpis['peak_production']:.2f}")
    logging.info(f"Efficiency: {kpis['efficiency']:.2%}")
    logging.info(f"Volatility: {kpis['volatility']:.4f}")
    
    if config['monitoring']['check_efficiency'] and kpis['efficiency'] < config['monitoring']['alert_threshold']:
        logging.info(f"\n⚠️  Alert: Production efficiency below threshold ({kpis['efficiency']:.2%} < {config['monitoring']['alert_threshold']:.2%})")
    
    plot_production_monitoring(df, config['data']['production_column'],
                             "Digital Integration Oil Production Monitoring",
                             output_dir / 'production_monitoring.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

