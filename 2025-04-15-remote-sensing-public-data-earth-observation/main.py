#!/usr/bin/env python3
"""
Remote Sensing and Public Data Earth Observation

Main entry point for running remote sensing analysis.
"""

import argparse
import yaml
import logging
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
    parser = argparse.ArgumentParser(description='Remote Sensing and Earth Observation')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df['date'] = pd.to_datetime(df['date'])
        band_cols = [col for col in df.columns if col not in ['date']]
    elif config['data']['generate_synthetic']:
                df = simulate_satellite_data(config['data']['n_points'], config['data']['n_bands'],
                                     config['data']['seed'])
        band_cols = [col for col in df.columns if col != 'date']
    else:
        raise ValueError("No data source specified")
    
        analysis = analyze_earth_observation(df, band_cols)
    
    logging.info(f"\nEarth Observation Analysis:")
    logging.info(f"Number of observations: {analysis['n_observations']}")
    logging.info(f"Number of bands: {analysis['n_bands']}")
    
    if config['analysis']['calculate_ndvi'] and config['analysis']['red_band'] in df.columns and config['analysis']['nir_band'] in df.columns:
                ndvi = calculate_ndvi(df, config['analysis']['red_band'], config['analysis']['nir_band'])
        logging.info(f"Mean NDVI: {ndvi.mean():.4f}")
        logging.info(f"NDVI Range: [{ndvi.min():.4f}, {ndvi.max():.4f}]")
    
    plot_earth_observation(df, band_cols, "Remote Sensing Earth Observation",
                          output_dir / 'earth_observation.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

