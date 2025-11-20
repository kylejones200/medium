#!/usr/bin/env python3
"""
Causal Inference in Time Series Econometrics

Main entry point for running causal inference analysis.

Usage:
    python main.py
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import perform_granger_causality_test, difference_in_differences, plot_causal_effect


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Causal Inference in Time Series')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        x = np.cumsum(np.random.normal(0, 1, config['data']['n_periods']))
        y = 0.5 * x + np.random.normal(0, 0.5, config['data']['n_periods'])
        data = pd.DataFrame({'x': x, 'y': y})
    else:
        raise ValueError("No data source specified")
    
    if config['analysis']['granger_causality']['enabled']:
        print("Performing Granger causality test...")
        result = perform_granger_causality_test(
            data, 'x', 'y', config['analysis']['granger_causality']['maxlag']
        )
        print("Granger causality test completed")
    
    if config['analysis']['difference_in_differences']['enabled']:
        print("Calculating difference-in-differences...")
        y_treated = np.cumsum(np.random.normal(0.1, 1, config['data']['n_periods']))
        y_control = np.cumsum(np.random.normal(0, 1, config['data']['n_periods']))
        did = difference_in_differences(
            y_treated, y_control, config['analysis']['difference_in_differences']['treatment_period']
        )
        print(f"Difference-in-differences estimate: {did:.4f}")
        plot_causal_effect(y_treated, y_control,
                          config['analysis']['difference_in_differences']['treatment_period'],
                          output_dir / 'causal_effect.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

