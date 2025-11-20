#!/usr/bin/env python3
"""
Ito's Lemma in Stochastic Finance

Main entry point for running stochastic process simulations.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from src.core import (
    ito_log_gbm,
    simulate_ou,
    simulate_standard_normal_from_bm,
    steady_state_ou_pdf,
    plot_gbm_simulation,
    plot_ou_process,
    plot_standard_normal,
    plot_ou_steady_state
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Ito's Lemma and Stochastic Processes")
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Simulating Geometric Brownian Motion...")
    t, S, ln_S = ito_log_gbm(
        config['simulation']['gbm']['S0'],
        config['simulation']['gbm']['mu'],
        config['simulation']['gbm']['sigma'],
        config['simulation']['gbm']['T'],
        config['simulation']['gbm']['steps'],
        config['simulation']['gbm']['seed']
    )
    plot_gbm_simulation(t, S, output_dir / 'ito_gbm_logprice.png')
    
    print("Simulating Ornstein-Uhlenbeck Process...")
    t_ou, r = simulate_ou(
        config['simulation']['ou']['r0'],
        config['simulation']['ou']['mu'],
        config['simulation']['ou']['theta'],
        config['simulation']['ou']['sigma'],
        config['simulation']['ou']['T'],
        config['simulation']['ou']['steps'],
        config['simulation']['ou']['seed']
    )
    plot_ou_process(t_ou, r, output_dir / 'ou_process.png')
    
    print("Generating standard normal from Brownian increments...")
    Z = simulate_standard_normal_from_bm(
        config['simulation']['standard_normal']['T'],
        config['simulation']['standard_normal']['steps'],
        config['simulation']['standard_normal']['seed']
    )
    plot_standard_normal(Z, output_dir / 'standard_normal.png')
    
    print("Computing steady-state distribution...")
    x_vals = np.linspace(
        config['simulation']['steady_state']['x_min'],
        config['simulation']['steady_state']['x_max'],
        config['simulation']['steady_state']['n_points']
    )
    pdf_vals = steady_state_ou_pdf(
        x_vals,
        config['simulation']['steady_state']['mu'],
        config['simulation']['steady_state']['theta'],
        config['simulation']['steady_state']['sigma']
    )
    plot_ou_steady_state(x_vals, pdf_vals, output_dir / 'ou_steady_state.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

