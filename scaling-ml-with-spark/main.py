#!/usr/bin/env python3
"""
Scaling Machine Learning Workflows by Moving from Python to Spark

Main entry point for running Spark ML scaling analysis.
"""

import argparse
import yaml
import logging
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
    parser = argparse.ArgumentParser(description='Scaling ML with Spark')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
    elif config['data']['source'] and Path(config['data']['source']).exists():
        df = pd.read_csv(config['data']['source'])
    elif config['data']['generate_synthetic']:
        logging.info("Simulating large dataset...")
        df = simulate_large_dataset(config['data']['n_samples'],
                                    config['data']['n_features'],
                                    config['data']['seed'])
    else:
        raise ValueError("No data source specified")
    
    logging.info("Analyzing Spark workflow...")
    analysis = analyze_spark_workflow(df)
    
    logging.info("Dataset Analysis:")
    logging.info(f"Number of samples: {analysis['n_samples']:,}")
    logging.info(f"Number of features: {analysis['n_features']}")
    logging.info(f"Memory usage: {analysis['memory_usage_mb']:.2f} MB")
    logging.info(f"Suitable for Spark: {analysis['spark_suitable']}")
    
    if config['spark']['use_spark']:
        logging.info("Note: Full Spark implementation would use:")
        logging.info("  from pyspark.sql import SparkSession")
        logging.info("  from pyspark.ml.feature import VectorAssembler")
        logging.info("  from pyspark.ml.classification import LogisticRegression")
        logging.info("  spark = SparkSession.builder.appName('MLPipeline').getOrCreate()")
    
    plot_spark_analysis(df, "Spark ML Scaling Analysis", output_dir / 'spark_analysis.png')
    
    logging.info(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

