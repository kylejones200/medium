#!/usr/bin/env python3
"""
Regression Discontinuity (RD) Analysis

Main entry point for running regression discontinuity analysis.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    fetch_fred_data,
    compute_inflation,
    create_rd_variables,
    fit_rd_model,
    fit_placebo_model,
    plot_rd_design,
    create_summary_table
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Regression Discontinuity Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    print("Fetching data from FRED...")
    unrate = fetch_fred_data({'UNRATE': config['data']['series']['UNRATE']},
                            config['data']['start_date'], config['data']['end_date'])
    civpart = fetch_fred_data({'CIVPART': config['data']['series']['CIVPART']},
                             config['data']['start_date'], config['data']['end_date'])
    cpi = fetch_fred_data({'CPIAUCSL': config['data']['series']['CPIAUCSL']},
                         config['data']['cpi_start_date'], config['data']['end_date'])
    
    print("Computing inflation...")
    inflation = compute_inflation(cpi)
    
    data = unrate.join(civpart).join(inflation)
    data.columns = ['UNRATE', 'CIVPART', 'Inflation']
    data = data.dropna().reset_index()
    
    print("Creating RD variables...")
    data = create_rd_variables(data, config['analysis']['cutoff_date'],
                              bandwidth=config['analysis']['bandwidth'])
    
    models = {}
    treatment_params = {}
    
    print("\nFitting bandwidth sensitivity models...")
    for bw in config['analysis']['bandwidth_sensitivity']:
        model = fit_rd_model(data, 'UNRATE', bw)
        models[f'Bandwidth ±{bw}'] = model
        treatment_params[f'Bandwidth ±{bw}'] = 'Treatment'
    
    print("Fitting placebo model...")
    placebo_model = fit_placebo_model(data, 'UNRATE', config['analysis']['placebo_cutoff'])
    models['Placebo Jan 2019'] = placebo_model
    treatment_params['Placebo Jan 2019'] = 'Placebo_Treatment'
    
    print("Fitting covariate continuity models...")
    inf_model = fit_rd_model(data, 'Inflation', config['analysis']['bandwidth'])
    lab_model = fit_rd_model(data, 'CIVPART', config['analysis']['bandwidth'])
    
    models['Inflation Check'] = inf_model
    models['Labor Participation Check'] = lab_model
    treatment_params['Inflation Check'] = 'Treatment'
    treatment_params['Labor Participation Check'] = 'Treatment'
    
    summary_table = create_summary_table(models, treatment_params)
    print("\nSummary Table:")
    print(summary_table)
    
    print("\nGenerating RD plots...")
    for outcome in config['analysis']['outcomes']:
        plot_rd_design(
            data['Months'],
            data[outcome['name']],
            outcome['description'] + ' Around ' + config['analysis']['cutoff_date'],
            outcome['ylabel'],
            output_dir / f'rd_plot_{outcome["name"].lower()}.png',
            outcome.get('ylim_zero', False)
        )
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

