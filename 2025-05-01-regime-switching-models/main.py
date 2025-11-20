#!/usr/bin/env python3
"""
Regime Switching Models for Time Series

Main entry point for running regime switching analysis.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
import pandas as pd
from pathlib import Path
from src.core import (
    generate_regime_data,
    fit_markov_switching,
    add_predictions,
    calculate_accuracy,
    calculate_regime_statistics,
    calculate_regime_durations,
    plot_regime_data,
    plot_regime_comparison,
    plot_density_distribution,
    plot_transition_matrix,
    plot_confusion_matrix
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Regime Switching Models')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Generating regime switching data...")
    df = generate_regime_data(
        config['data']['n_samples'],
        tuple(config['data']['regime_probs']),
        tuple(config['data']['stds']),
        config['data']['seed']
    )
    
    print("Fitting Markov switching model...")
    result = fit_markov_switching(
        df['Data'].values,
        config['model']['k_regimes'],
        config['model']['switching_variance']
    )
    
    print(result.summary())
    print("\nTransition Matrix:")
    print(result.regime_transition)
    
    df = add_predictions(df, result)
    
    accuracy = calculate_accuracy(df)
    print(f"\nPrediction Accuracy: {accuracy:.2%}")
    
    regime_stats = calculate_regime_statistics(df)
    print("\nRegime Statistics:")
    for regime, stats_dict in regime_stats.items():
        print(f"\nRegime {regime}:")
        print(f"Mean: {stats_dict['mean']:.2f}")
        print(f"Std: {stats_dict['std']:.2f}")
        print(f"Skewness: {stats_dict['skewness']:.2f}")
        print(f"Kurtosis: {stats_dict['kurtosis']:.2f}")
    
    durations = calculate_regime_durations(df)
    print("\nAverage Duration in Each Regime:")
    for regime, duration in durations.items():
        print(f"Regime {regime}: {duration:.2f} periods")
    
    transitions = pd.DataFrame({
        'From': df['Predicted_Regime'][:-1],
        'To': df['Predicted_Regime'][1:]
    })
    print("\nTransition Counts:")
    print(pd.crosstab(transitions['From'], transitions['To']))
    
    if config['analysis']['run_all_plots']:
        print("\nGenerating plots...")
        plot_regime_data(df, output_dir / 'original_data_regimes.png')
        plot_regime_comparison(df, output_dir / 'true_vs_predicted_regimes.png')
        plot_density_distribution(df, output_dir / 'density_distribution.png')
        plot_transition_matrix(result, output_dir / 'transition_matrix.png')
        plot_confusion_matrix(df, output_dir / 'confusion_matrix.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

