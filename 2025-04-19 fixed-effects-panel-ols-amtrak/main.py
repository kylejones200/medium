#!/usr/bin/env python3
"""
Fixed Effects Time Series Modeling with Panel OLS

Main entry point for running panel OLS analysis with fixed effects.

Usage:
    python main.py
    python main.py --data-path data/panel_data.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import prepare_panel_data, fit_panel_ols, plot_panel_results


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Fixed Effects Panel OLS Modeling')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_path = args.data_path if args.data_path else Path(config['data']['source'])
    
    if data_path.exists():
        df = pd.read_csv(data_path)
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        entities = [f"Entity_{i}" for i in range(config['data']['n_entities'])]
        periods = range(2020, 2020 + config['data']['n_periods'])
        
        data = []
        for entity in entities:
            base = np.random.normal(100, 20)
            for period in periods:
                value = base + np.random.normal(0, 5) + (period - 2020) * 2
                data.append({'Station': entity, 'Year': period, 'Ridership': max(0, value)})
        df = pd.DataFrame(data)
    else:
        raise ValueError("No data source specified")
    
    print("Preparing panel data...")
    df = prepare_panel_data(df, config['data']['entity_column'], 
                           config['data']['time_column'], config['data']['value_column'])
    
    print(f"\nPanel Data Shape: {df.shape}")
    print(f"Entities: {df[config['data']['entity_column']].nunique()}")
    print(f"Time Periods: {df[config['data']['time_column']].nunique()}")
    
    if config['model']['entity_fixed_effects'] or config['model']['time_fixed_effects']:
        print("\nFitting panel OLS with fixed effects...")
        model = fit_panel_ols(df, config['data']['value_column'], config['model']['features'],
                            config['data']['entity_column'] if config['model']['entity_fixed_effects'] else None,
                            config['data']['time_column'] if config['model']['time_fixed_effects'] else None)
        
        print("\nModel Summary:")
        print(model.summary())
    
    plot_panel_results(df, config['data']['entity_column'], config['data']['value_column'],
                      "Panel Data: Fixed Effects Analysis", output_dir / 'panel_analysis.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

