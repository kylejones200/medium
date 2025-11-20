#!/usr/bin/env python3
"""
Seasonal and Trend Decomposition Methods for Time Series

Main entry point for running decomposition analysis.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    generate_synthetic_data,
    decompose_additive,
    decompose_multiplicative,
    robust_decomposition,
    analyze_components,
    plot_decomposition_comparison
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Seasonal and Trend Decomposition')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Generating synthetic data...")
    df = generate_synthetic_data(
        config['data']['start_date'],
        config['data']['end_date'],
        config['data']['frequency'],
        config['data']['seed']
    )
    
    additive_decomp = None
    mult_decomp = None
    
    if config['analysis']['run_additive']:
        print("Running additive decomposition...")
        additive_decomp = decompose_additive(df, config['decomposition']['period'])
    
    if config['analysis']['run_multiplicative']:
        print("Running multiplicative decomposition...")
        mult_decomp = decompose_multiplicative(df, config['decomposition']['period'])
    
    if config['analysis']['run_comparison'] and additive_decomp and mult_decomp:
        print("Generating comparison plot...")
        plot_decomposition_comparison(df, additive_decomp, mult_decomp,
                                    output_dir / 'compare_decomposition_methods.png')
    
    if config['analysis']['run_robust']:
        print("Running robust decomposition...")
        robust_results = robust_decomposition(df, config['decomposition']['period'])
        print("Robust decomposition completed")
    
    if additive_decomp:
        analysis_results = analyze_components(additive_decomp)
        print("\nComponent Analysis:")
        print(f"Trend Direction: {analysis_results['trend_direction']}")
        print(f"Trend Strength: {analysis_results['trend_strength']:.4f}")
        print(f"Seasonal Amplitude: {analysis_results['seasonal_amplitude']:.4f}")
        print(f"Residual Variance: {analysis_results['residual_variance']:.4f}")
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

