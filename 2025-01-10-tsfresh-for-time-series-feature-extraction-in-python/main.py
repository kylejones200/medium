#!/usr/bin/env python3
"""
TSFresh for Time Series Feature Extraction

Main entry point for running TSFresh feature extraction.
"""

import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from src.core import (
    generate_time_series_data,
    extract_tsfresh_features,
    select_relevant_features,
)
from tsfresh.feature_extraction import ComprehensiveFCParameters

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='TSFresh Feature Extraction')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        logging.info(f"Loading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
    else:
                df = generate_time_series_data(
            config['data']['n_series'],
            config['data']['n_timesteps'],
            config['data']['seed']
        )
    
    plot_time_series(df, output_dir / 'time_series_sample.png',
                    config['output']['n_series_to_plot'])
    
        fc_parameters = ComprehensiveFCParameters() if config['feature_extraction']['use_comprehensive'] else None
    
    extracted_features = extract_tsfresh_features(
        df,
        config['data']['column_id'],
        config['data']['column_sort'],
        config['data']['column_value'],
        fc_parameters
    )
    
    logging.info(f"Extracted {len(extracted_features.columns)} features")
    logging.info(f"Feature names: {list(extracted_features.columns[:10])}...")
    
    if config['feature_extraction']['feature_selection']:
                y = pd.Series(np.random.randint(0, 2, size=len(extracted_features)))
        selected_features = select_relevant_features(
            extracted_features,
            y,
            config['feature_extraction']['fdr_level']
        )
        logging.info(f"Selected {len(selected_features.columns)} relevant features")
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

