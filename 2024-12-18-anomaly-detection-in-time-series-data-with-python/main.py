#!/usr/bin/env python3
"""
Anomaly Detection in Time Series Data with Python

Main entry point for running anomaly detection analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    create_lagged_features,
    detect_anomalies_isolation_forest,
    detect_anomalies_statistical,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection in Time Series')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        data = df.iloc[:, 0]
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='D')
        base = np.sin(np.arange(config['data']['n_periods']) / 30) * 10 + 50
        anomalies = np.random.choice(config['data']['n_periods'], size=10, replace=False)
        values = base + np.random.normal(0, 2, config['data']['n_periods'])
        values[anomalies] += np.random.normal(0, 15, len(anomalies))
        data = pd.Series(values, index=dates)
    else:
        raise ValueError("No data source specified")
    
    if config['model']['method'] == 'isolation_forest':
                features_df = create_lagged_features(data, config['model']['lag'])
        X = features_df.values
        
                anomalies = detect_anomalies_isolation_forest(
            X, config['model']['contamination'], config['data']['seed']
        )
        anomalies_full = np.zeros(len(data))
        anomalies_full[features_df.index] = anomalies
    elif config['model']['method'] == 'statistical':
                anomalies_full = detect_anomalies_statistical(data, config['model']['statistical_threshold'])
    else:
        raise ValueError(f"Unknown method: {config['model']['method']}")
    
    n_anomalies = anomalies_full.sum()
    logging.info(f"\nDetected {n_anomalies} anomalies ({n_anomalies/len(data)*100:.2f}%)")
    
    plot_anomalies(data, anomalies_full, "Anomaly Detection in Time Series",
                  output_dir / 'anomaly_detection.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

