#!/usr/bin/env python3
"""
Matrix Factorization for Long-Term Events (MFLE) for Time Series

Main entry point for running matrix factorization analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_multivariate_series,
    apply_svd,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Matrix Factorization for Time Series')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
        data, time = generate_multivariate_series(
        config['data']['n_series'],
        config['data']['n_timesteps'],
        config['data']['time_end'],
        config['data']['noise_std'],
        config['data']['seed']
    )
    
        svd, latent_features, reconstructed = apply_svd(data, config['model']['n_components'])
    
    logging.info(f"Explained variance ratio: {svd.explained_variance_ratio_}")
    logging.info(f"Total explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    
    plot_reconstruction_comparison(
        data,
        reconstructed,
        time,
        config['output']['n_series_to_plot'],
        output_dir / 'reconstruction_comparison.png'
    )
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

