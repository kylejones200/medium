#!/usr/bin/env python3
"""
Finding the Change Point in Competitive Hot Dog Eating with Bayesian Analysis

Main entry point for running Bayesian change point detection.
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
    parser = argparse.ArgumentParser(description='Bayesian Change Point Detection')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
    elif config['data']['generate_synthetic']:
                df = simulate_change_point_data(
            config['data']['n_periods'],
            config['data']['change_point'],
            config['data']['before_mean'],
            config['data']['after_mean'],
            config['data']['std'],
            config['data']['seed']
        )
    else:
        raise ValueError("No data source specified")
    
    logging.info(f"Detecting change point using {config['model']['method']}...")
    detected_cp = detect_change_point_basic(df['value'].values, 
                                           config['model']['detection_window'])
    
    logging.info(f"\nDetected change point: {detected_cp}")
    logging.info(f"True change point: {config['data']['change_point']}")
    logging.error(f": {abs(detected_cp - config['data']['change_point'])} periods")
    
                    plot_change_point_detection(df, detected_cp, "Change Point Detection",
                               output_dir / 'change_point.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

