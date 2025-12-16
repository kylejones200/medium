#!/usr/bin/env python3
"""
Using Arrow to Wrangle Dates and Times in Python

Main entry point for demonstrating Arrow date/time operations.
"""

import argparse
import yaml
import logging
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
    
        results = demonstrate_arrow_operations()
    
        logging.info(f"Current UTC: {results['current_utc']}")
    logging.info(f"2 hours ago: {results['two_hours_ago']}")
    logging.info(f"Next week: {results['next_week']}")
    logging.info(f"US Central Time: {results['us_central']}")
    logging.info(f"Humanized: {results['humanized']}")
    logging.info(f"Custom Format: {results['formatted']}")
    logging.info(f"Parsed Time: {results['parsed']}")
    logging.info(f"Rounded (floor hour): {results['rounded']}")
    logging.info(f"Interval Duration: {results['interval_hours']:.2f} hours")
    
        times = [results['current_utc'].shift(days=-i) for i in range(5)]
    df = create_time_series_dataframe(times)
    logging.info(df)
    
    if __name__ == "__main__":
    from src.core import create_time_series_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

