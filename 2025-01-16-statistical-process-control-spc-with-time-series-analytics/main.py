#!/usr/bin/env python3
"""
Statistical Process Control (SPC) with Time Series Analytics

Main entry point for running SPC control chart analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_process_data,
    calculate_control_limits,
    identify_out_of_control,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Statistical Process Control')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
        df = generate_process_data(
        config['data']['start_date'],
        config['data']['periods'],
        config['data']['frequency'],
        config['data']['mean'],
        config['data']['std'],
        config['data']['seed']
    )
    
        limits = calculate_control_limits(df, config['control_limits']['sigma_multiplier'])
    
    logging.info(f"Mean: {limits['mean']:.2f}")
    logging.info(f"Standard Deviation: {limits['std_dev']:.2f}")
    logging.info(f"UCL: {limits['ucl']:.2f}")
    logging.info(f"LCL: {limits['lcl']:.2f}")
    
    out_of_control = identify_out_of_control(df, limits)
    n_outliers = out_of_control.sum()
    logging.info(f"Out-of-control points: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    
    plot_control_chart(df, limits, out_of_control, output_dir / 'control_chart.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

