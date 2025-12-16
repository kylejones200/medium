#!/usr/bin/env python3
"""
Integrating Excel and Python for Business Analytics

Main entry point for running Excel-Python integration analysis.
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
    parser = argparse.ArgumentParser(description='Integrating Excel and Python for Business Analytics')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--excel-path', type=Path, default=None, help='Path to Excel file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.excel_path and args.excel_path.exists():
        df = read_excel_data(args.excel_path, config['data']['sheet_name'])
    elif config['data']['source'] and Path(config['data']['source']).exists():
        df = read_excel_data(Path(config['data']['source']), config['data']['sheet_name'])
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_rows'], freq='D')
        values = 100 + 10 * np.sin(np.arange(config['data']['n_rows']) / 10) + np.random.normal(0, 5, config['data']['n_rows'])
        df = pd.DataFrame({
            'Date': dates,
            'Value': values,
            'Category': np.random.choice(['A', 'B', 'C'], config['data']['n_rows'])
        })
    else:
        raise ValueError("No data source specified")
    
    logging.info("Analyzing Excel data...")
    analysis = analyze_excel_data(df)
    
    logging.info("Excel Data Analysis:")
    logging.info(f"Shape: {analysis['shape']}")
    logging.info(f"Columns: {', '.join(analysis['columns'])}")
    
    if analysis['summary']:
        logging.info("Summary Statistics:")
        logging.info(f"\n{pd.DataFrame(analysis['summary'])}")
    
    output_excel = output_dir / config['output']['excel_output']
    write_excel_data(df, output_excel)
    logging.info(f"Data written to {output_excel}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        plot_excel_analysis(df, numeric_cols[0], "Excel Data Analysis",
                           output_dir / 'excel_analysis.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

