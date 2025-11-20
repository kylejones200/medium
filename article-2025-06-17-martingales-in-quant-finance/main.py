#!/usr/bin/env python3
"""
Martingales in Quantitative Finance

Main entry point for running martingale simulations.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    simulate_binomial_tree,
    simulate_exponential_martingale,
    simulate_girsanov,
    plot_binomial_path,
    plot_exponential_martingale,
    plot_girsanov_transformation
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Martingales in Quantitative Finance")
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Simulating binomial tree...")
    binomial_path = simulate_binomial_tree(
        config['simulation']['binomial']['S0'],
        config['simulation']['binomial']['u'],
        config['simulation']['binomial']['d'],
        config['simulation']['binomial']['r'],
        config['simulation']['binomial']['steps'],
        config['simulation']['binomial']['seed']
    )
    plot_binomial_path(binomial_path, output_dir / 'binomial_path.png')
    
    print("Simulating exponential martingale...")
    t, W, Z = simulate_exponential_martingale(
        config['simulation']['exponential_martingale']['theta'],
        config['simulation']['exponential_martingale']['T'],
        config['simulation']['exponential_martingale']['steps'],
        config['simulation']['exponential_martingale']['seed']
    )
    plot_exponential_martingale(t, Z, output_dir / 'exponential_martingale.png')
    
    print("Simulating Girsanov transformation...")
    t, W, W_tilde = simulate_girsanov(
        config['simulation']['girsanov']['theta'],
        config['simulation']['girsanov']['T'],
        config['simulation']['girsanov']['steps'],
        config['simulation']['girsanov']['seed']
    )
    plot_girsanov_transformation(t, W, W_tilde, output_dir / 'girsanov_shift.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

