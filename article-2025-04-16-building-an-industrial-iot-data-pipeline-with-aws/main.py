#!/usr/bin/env python3
"""
Building an Industrial IoT Data Pipeline with AWS

Main entry point for running IoT data pipeline analysis.

Usage:
    python main.py
    python main.py --data-path data/iot_data.csv
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.core import simulate_iot_data, analyze_sensor_data, plot_sensor_data


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
        print("Simulating IoT sensor data...")
        df = simulate_iot_data(config['data']['n_points'], config['data']['n_sensors'],
                              config['data']['seed'])
        sensor_cols = [col for col in df.columns if col != 'timestamp']
    else:
        raise ValueError("No data source specified")
    
    print("Analyzing sensor data...")
    analysis = analyze_sensor_data(df, sensor_cols)
    
    print(f"\nSensor Data Analysis:")
    print(f"Number of samples: {analysis['n_samples']}")
    print(f"Number of sensors: {analysis['n_sensors']}")
    print(f"\nMean values:")
    for sensor, mean_val in analysis['mean_values'].items():
        print(f"  {sensor}: {mean_val:.2f}")
    
    print(f"\nAWS Services: {', '.join(config['aws']['services'])}")
    print("Note: Full AWS pipeline would integrate:")
    print("  - IoT Core: Device connectivity")
    print("  - Kinesis: Real-time data streaming")
    print("  - S3: Data storage")
    print("  - Lambda: Serverless processing")
    print("  - DynamoDB: Time series storage")
    
    plot_sensor_data(df, sensor_cols, "Industrial IoT Sensor Data",
                    output_dir / 'sensor_data.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

