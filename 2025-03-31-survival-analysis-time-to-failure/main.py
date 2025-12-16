#!/usr/bin/env python3
"""
Survival Analysis Time to Failure

Main entry point for running survival analysis.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    load_survival_data,
    fit_kaplan_meier,
    fit_weibull_survival,
    fit_cox_proportional_hazards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Survival Analysis Time to Failure')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = load_survival_data(args.data_path)
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        durations = np.random.exponential(scale=100, size=config['data']['n_samples'])
        events = np.random.binomial(1, 0.7, size=config['data']['n_samples'])
        df = pd.DataFrame({
            config['data']['duration_column']: durations,
            config['data']['event_column']: events
        })
    else:
        raise ValueError("No data source specified")
    
    if config['model']['kaplan_meier']:
                kmf = fit_kaplan_meier(df, config['data']['duration_column'], config['data']['event_column'])
        logging.info(f"Median survival time: {kmf.median_survival_time_:.2f}")
        plot_survival_curve(kmf, "Kaplan-Meier Survival Curve", output_dir / 'kaplan_meier.png')
    
    if config['model']['weibull']:
                wf = fit_weibull_survival(df, config['data']['duration_column'], config['data']['event_column'])
        logging.info(wf.summary)
    
    if config['model']['cox_ph'] and config['model']['covariates']:
                cph = fit_cox_proportional_hazards(
            df, config['data']['duration_column'], config['data']['event_column'],
            config['model']['covariates']
        )
        logging.info(cph.summary)
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

