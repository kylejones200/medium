#!/usr/bin/env python3
"""
Statistical Process Control (SPC) with Time Series Analytics

Main entry point for running SPC control chart analysis.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    generate_process_data,
    calculate_control_limits,
    identify_out_of_control,
    plot_control_chart
)


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
    
    print("Generating process data...")
    df = generate_process_data(
        config['data']['start_date'],
        config['data']['periods'],
        config['data']['frequency'],
        config['data']['mean'],
        config['data']['std'],
        config['data']['seed']
    )
    
    print("Calculating control limits...")
    limits = calculate_control_limits(df, config['control_limits']['sigma_multiplier'])
    
    print(f"Mean: {limits['mean']:.2f}")
    print(f"Standard Deviation: {limits['std_dev']:.2f}")
    print(f"UCL: {limits['ucl']:.2f}")
    print(f"LCL: {limits['lcl']:.2f}")
    
    out_of_control = identify_out_of_control(df, limits)
    n_outliers = out_of_control.sum()
    print(f"Out-of-control points: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    
    plot_control_chart(df, limits, out_of_control, output_dir / 'control_chart.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

