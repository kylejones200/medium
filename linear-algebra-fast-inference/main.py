#!/usr/bin/env python3
"""
Linear Algebra for Fast Inference

Main entry point for running linear algebra inference analysis.
"""

import argparse
import yaml
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Linear Algebra for Fast Inference')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    logging.info("Creating linear system...")
    A, b, x_true = create_linear_system(config['system']['n_variables'],
                                        config['system']['m_equations'],
                                        config['system']['seed'])
    
    logging.info(f"System dimensions: {A.shape}")
    logging.info(f"Solving using {config['inference']['method']} method...")
    x_pred = solve_linear_system(A, b, config['inference']['method'])
    
    logging.info("Calculating inference metrics...")
    metrics = calculate_inference_metrics(x_true, x_pred)
    
    logging.info("Inference Metrics:")
    logging.info(f"MSE: {metrics['mse']:.6f}")
    logging.info(f"MAE: {metrics['mae']:.6f}")
    logging.info(f"Correlation: {metrics['correlation']:.4f}")
    
    plot_inference_results(x_true, x_pred, "Linear Algebra Fast Inference",
                          output_dir / 'inference_results.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

