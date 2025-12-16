#!/usr/bin/env python3
"""
Asset Price Modeling

Main entry point for running asset price simulations.
"""

import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    simulate_wiener_process,
    simulate_gbm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Asset Price Modeling")
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
        W = simulate_wiener_process(
        config['simulation']['wiener']['n_steps'],
        config['simulation']['wiener']['T'],
        config['simulation']['wiener']['seed']
    )
    t_W = np.linspace(0, config['simulation']['wiener']['T'], len(W))
    plot_wiener_process(W, t_W, output_dir / 'wiener_process.png')
    
        t, S = simulate_gbm(
        config['simulation']['gbm']['S0'],
        config['simulation']['gbm']['mu'],
        config['simulation']['gbm']['sigma'],
        config['simulation']['gbm']['T'],
        config['simulation']['gbm']['n_steps'],
        config['simulation']['gbm']['seed']
    )
    plot_gbm_simulation(t, S, output_dir / 'gbm_simulation.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

