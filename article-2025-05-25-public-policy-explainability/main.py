#!/usr/bin/env python3
"""
Public Policy Explainability with SHAP

Main entry point for running SHAP explainability analysis.

Usage:
    python main.py --data-path data/public_health_data.csv
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    load_data,
    prepare_data,
    train_model,
    compute_shap_values,
    save_shap_plots
)


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Public Policy Explainability')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data_path = args.data_path if args.data_path else Path(config['data']['source'])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print("Loading data...")
    X, y = load_data(data_path)
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        X, y,
        config['model']['test_size'],
        config['model']['random_state']
    )
    
    print("Training model...")
    model = train_model(
        X_train, y_train,
        config['model']['n_estimators'],
        config['model']['random_state']
    )
    
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    if config['shap']['enabled']:
        print("Computing SHAP values...")
        explainer, shap_values = compute_shap_values(model, X_test, X.columns.tolist())
        
        if config['shap']['plot_summary'] or config['shap']['plot_bar']:
            save_shap_plots(explainer, shap_values, X_test, X.columns.tolist(), output_dir)
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

