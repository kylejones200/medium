#!/usr/bin/env python3
"""
PCA and Mahalanobis Distance for Predictive Maintenance

Main entry point for running predictive maintenance analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    load_cmapss_data,
    apply_pca,
    calculate_mahalanobis_distance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='PCA and Mahalanobis Distance for Predictive Maintenance')
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
    
        df = load_cmapss_data(data_path, config['data']['separator'])
    
        df, pca, scaler = apply_pca(df, config['model']['selected_sensors'], 
                               config['model']['n_components'])
    
    logging.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logging.info(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
        pca_columns = [f'pca_{i+1}' for i in range(config['model']['n_components'])]
    df['distance'] = calculate_mahalanobis_distance(df, pca_columns)
    
    logging.info(f"Plotting health indices for {len(config['analysis']['units_to_plot'])} units...")
    for unit_id in config['analysis']['units_to_plot']:
        if unit_id in df['unit'].values:
            plot_health_index(df, unit_id, 'distance', config['model']['threshold'],
                            output_dir / f'health_index_unit_{unit_id}.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

