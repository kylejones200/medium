#!/usr/bin/env python3
"""
Building an Industrial IoT Data Pipeline with AWS

Main entry point for running IoT data pipeline analysis.
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
    parser = argparse.ArgumentParser(description='Industrial IoT Data Pipeline with AWS')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        sensor_cols = [col for col in df.columns if 'sensor' in col.lower()]
    elif config['data']['generate_synthetic']:
                df = simulate_iot_data(config['data']['n_points'], config['data']['n_sensors'],
                              config['data']['seed'])
        sensor_cols = [col for col in df.columns if col != 'timestamp']
    else:
        raise ValueError("No data source specified")
    
        analysis = analyze_sensor_data(df, sensor_cols)
    
    logging.info(f"\nSensor Data Analysis:")
    logging.info(f"Number of samples: {analysis['n_samples']}")
    logging.info(f"Number of sensors: {analysis['n_sensors']}")
    logging.info(f"\nMean values:")
    for sensor, mean_val in analysis['mean_values'].items():
        logging.info(f"  {sensor}: {mean_val:.2f}")
    
    logging.info(f"\nAWS Services: {', '.join(config['aws']['services'])}")
                            plot_sensor_data(df, sensor_cols, "Industrial IoT Sensor Data",
                    output_dir / 'sensor_data.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

