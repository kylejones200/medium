#!/usr/bin/env python3
"""
DTW (Dynamic Time Warping) Analysis of Amtrak Ridership

Main entry point for running DTW analysis on Amtrak ridership patterns.

Usage:
    python main.py
    python main.py --data-path data/ridership_data.csv
"""

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import prepare_time_series_matrix, compute_dtw_distance_matrix, find_similar_stations, plot_dtw_matrix


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='DTW Analysis of Amtrak Ridership')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_path = args.data_path if args.data_path else Path(config['data']['source'])
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=['Unnamed: 0'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        stations = [f"Station_{i}" for i in range(config['data']['n_stations'])]
        periods = range(2010, 2010 + config['data']['n_periods'])
        
        data = []
        for station in stations:
            base = np.random.normal(1000, 200)
            trend = np.random.choice([-1, 0, 1])
            for period in periods:
                value = base + trend * (period - 2010) * 10 + np.random.normal(0, 50)
                data.append({'Station': station, 'Year': period, 'Ridership': max(0, value)})
        df = pd.DataFrame(data)
    else:
        raise ValueError("No data source specified")
    
    print("Preparing time series matrix...")
    ts_matrix = prepare_time_series_matrix(df, config['data']['station_column'],
                                         config['data']['time_column'], config['data']['value_column'])
    
    print(f"\nTime Series Matrix Shape: {ts_matrix.shape}")
    print(f"Stations: {len(ts_matrix.index)}")
    print(f"Time Periods: {len(ts_matrix.columns)}")
    
    print("\nComputing DTW distance matrix...")
    dtw_df = compute_dtw_distance_matrix(ts_matrix)
    
    if config['analysis']['save_dtw_matrix']:
        dtw_path = output_dir / 'dtw_matrix.csv'
        dtw_df.to_csv(dtw_path)
        print(f"\nDTW matrix saved to {dtw_path}")
    
    if config['analysis']['target_station'] in dtw_df.index:
        print(f"\nFinding similar stations to {config['analysis']['target_station']}...")
        similar = find_similar_stations(dtw_df, config['analysis']['target_station'],
                                       config['analysis']['n_similar'])
        print("\nMost Similar Stations:")
        for station, distance in similar.items():
            print(f"  {station}: {distance:.4f}")
    
    plot_dtw_matrix(dtw_df, "DTW Distance Matrix: Amtrak Ridership Patterns",
                   output_dir / 'dtw_matrix.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

