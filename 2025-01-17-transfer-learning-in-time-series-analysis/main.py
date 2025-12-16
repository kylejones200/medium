#!/usr/bin/env python3
"""
Transfer Learning in Time Series Analysis

Main entry point for running transfer learning analysis.
"""

import argparse
import yaml
import logging
import numpy as np
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
    parser = argparse.ArgumentParser(description='Transfer Learning in Time Series')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        source_data = pd.Series(np.sin(np.arange(config['data']['n_periods']) / 10) + np.random.normal(0, 0.1, config['data']['n_periods']))
        target_data = pd.Series(np.cos(np.arange(config['data']['n_periods']) / 10) + np.random.normal(0, 0.1, config['data']['n_periods']))
    else:
        raise ValueError("No data source specified")
    
        source_features = create_features(source_data, config['model']['lag'])
    target_features = create_features(target_data, config['model']['lag'])
    
    X_source = source_features.drop(columns=['target']).values
    y_source = source_features['target'].values
    X_target = target_features.drop(columns=['target']).values
    y_target = target_features['target'].values
    
        model = train_source_model(X_source, y_source)
    pred_source = model.predict(X_source)
    
    if config['model']['fine_tune']:
                model = fine_tune_model(model, X_target, y_target)
    
    pred_target = model.predict(X_target)
    
    plot_transfer_learning(y_source, y_target, pred_source, pred_target,
                          output_dir / 'transfer_learning.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

