#!/usr/bin/env python3
"""
Using Arrow to Wrangle Dates and Times in Python

Main entry point for demonstrating Arrow date/time operations.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import demonstrate_arrow_operations


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Arrow Date/Time Wrangling')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("Demonstrating Arrow operations...")
    results = demonstrate_arrow_operations()
    
    print("\nArrow Operations Results:")
    print(f"Current UTC: {results['current_utc']}")
    print(f"2 hours ago: {results['two_hours_ago']}")
    print(f"Next week: {results['next_week']}")
    print(f"US Central Time: {results['us_central']}")
    print(f"Humanized: {results['humanized']}")
    print(f"Custom Format: {results['formatted']}")
    print(f"Parsed Time: {results['parsed']}")
    print(f"Rounded (floor hour): {results['rounded']}")
    print(f"Interval Duration: {results['interval_hours']:.2f} hours")
    
    print("\nCreating time series DataFrame...")
    times = [results['current_utc'].shift(days=-i) for i in range(5)]
    df = create_time_series_dataframe(times)
    print(df)
    
    print("\nDemonstration complete.")


if __name__ == "__main__":
    from src.core import create_time_series_dataframe
    main()

