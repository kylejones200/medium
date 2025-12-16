#!/usr/bin/env python3
"""
Regime Switching Models for Time Series

Main entry point for running regime switching analysis.
"""

import argparse
import yaml
import logging
import pandas as pd
from pathlib import Path
from src.core import (
    generate_regime_data,
    fit_markov_switching,
    add_predictions,
    calculate_accuracy,
    calculate_regime_statistics,
    calculate_regime_durations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    
    logging.info("Generating regime switching data...")
    df = generate_regime_data(
        config['data']['n_samples'],
        tuple(config['data']['regime_probs']),
        tuple(config['data']['stds']),
        config['data']['seed']
    )
    
    logging.info("Fitting Markov switching model...")
    result = fit_markov_switching(
        df['Data'].values,
        config['model']['k_regimes'],
        config['model']['switching_variance']
    )
    
    logging.info(f"\n{result.summary()}")
    logging.info("Transition Matrix:")
    logging.info(f"\n{result.regime_transition}")
    
    df = add_predictions(df, result)
    
    accuracy = calculate_accuracy(df)
    logging.info(f"Prediction Accuracy: {accuracy:.2%}")
    
    regime_stats = calculate_regime_statistics(df)
    logging.info("Regime Statistics:")
    for regime, stats_dict in regime_stats.items():
        logging.info(f"Regime {regime}:")
        logging.info(f"Mean: {stats_dict['mean']:.2f}")
        logging.info(f"Std: {stats_dict['std']:.2f}")
        logging.info(f"Skewness: {stats_dict['skewness']:.2f}")
        logging.info(f"Kurtosis: {stats_dict['kurtosis']:.2f}")
    
    durations = calculate_regime_durations(df)
    logging.info("Average Duration in Each Regime:")
    for regime, duration in durations.items():
        logging.info(f"Regime {regime}: {duration:.2f} periods")
    
    transitions = pd.DataFrame({
        'From': df['Predicted_Regime'][:-1],
        'To': df['Predicted_Regime'][1:]
    })
    logging.info("Transition Counts:")
    logging.info(f"\n{pd.crosstab(transitions['From'], transitions['To'])}")
    
    if config['analysis']['run_all_plots']:
        logging.info("Generating plots...")
        plot_regime_data(df, output_dir / 'original_data_regimes.png')
        plot_regime_comparison(df, output_dir / 'true_vs_predicted_regimes.png')
        plot_density_distribution(df, output_dir / 'density_distribution.png')
        plot_transition_matrix(result, output_dir / 'transition_matrix.png')
        plot_confusion_matrix(df, output_dir / 'confusion_matrix.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()
