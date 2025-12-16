#!/usr/bin/env python3
"""
Aeon for Time Series Clustering with Python

Main entry point for running time series classification.
"""

import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_dataset,
    split_data,
    fit_classifier,
    evaluate_classifier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Time Series Clustering with aeon')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    X, y = generate_dataset(
        config['data']['n_instances'],
        config['data']['n_timepoints'],
        config['data']['seed']
    )
    
    plot_sample_series(X, ["Sine", "Cosine", "Sine2x"],
                      output_dir / 'sample_series.png')
    
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )
    
    clf = fit_classifier(X_train, y_train, config['model']['n_neighbors'])
    results = evaluate_classifier(clf, X_test, y_test)
    
    logging.info(f"Accuracy: {results['accuracy']:.2%}")
    logging.info(f"Data shape: {X.shape}")
    logging.info(f"Number of classes: {len(set(y))}")
    logging.info(f"Class distribution: {np.bincount(y)}")
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

