#!/usr/bin/env python3
"""
Conceptual Architecture for Supply Chain Management with AWS

Main entry point for running supply chain management analysis.
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
    parser = argparse.ArgumentParser(description='Conceptual Supply Chain Architecture with AWS')
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
        inventory_cols = [col for col in df.columns if 'inventory' in col.lower()]
    elif config['data']['generate_synthetic']:
                df = simulate_supply_chain_data(config['data']['n_points'], config['data']['n_nodes'],
                                       config['data']['seed'])
        inventory_cols = [col for col in df.columns if 'inventory' in col.lower()]
    else:
        raise ValueError("No data source specified")
    
        analysis = analyze_supply_chain(df, inventory_cols)
    
    logging.info(f"\nSupply Chain Analysis:")
    logging.info(f"Number of samples: {analysis['n_samples']}")
    logging.info(f"Number of nodes: {analysis['n_nodes']}")
    logging.info(f"Total average inventory: {analysis['total_inventory']:.2f}")
    
    logging.info(f"\nAWS Services: {', '.join(config['aws']['services'])}")
    
    plot_supply_chain(df, inventory_cols, "Supply Chain Management Architecture",
                     output_dir / 'supply_chain.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

