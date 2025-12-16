#!/usr/bin/env python3
"""
Working with Missing Data in Business Analytics

Main entry point for running missing data analysis.
"""

import argparse
import yaml
import logging
import numpy as np
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
    parser = argparse.ArgumentParser(description='Working with Missing Data in Business Analytics')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path:
        df = load_data_with_missing(file_path=args.data_path)
    elif config['data']['url']:
        df = load_data_with_missing(url=config['data']['url'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        df = load_data_with_missing()
    else:
        raise ValueError("No data source specified")
    
    logging.info("Analyzing missing data...")
    analysis = analyze_missing_data(df)
    
    logging.info("Missing Data Summary:")
    for col, count in analysis['missing_counts'].items():
        if count > 0:
            pct = analysis['missing_percentages'][col]
            logging.info(f"  {col}: {count} ({pct:.2f}%)")
    
    if analysis['total_missing'] > 0:
        logging.info(f"Imputing missing values using {config['imputation']['strategy']} strategy...")
        df_imputed = impute_missing_values(df, config['imputation']['strategy'])
        
        analysis_after = analyze_missing_data(df_imputed)
        logging.info(f"Missing values after imputation: {analysis_after['total_missing']}")
        
        df_imputed.to_csv(output_dir / 'imputed_output.csv', index=False)
        logging.info(f"Imputed data saved to {output_dir / 'imputed_output.csv'}")
    
    plot_missing_analysis(df, "Missing Data Analysis", output_dir / 'missing_analysis.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

