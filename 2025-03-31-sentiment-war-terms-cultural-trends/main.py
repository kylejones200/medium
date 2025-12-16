#!/usr/bin/env python3
"""
Sentiment Analysis: War Terms and Cultural Trends

Main entry point for running sentiment analysis.
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
    parser = argparse.ArgumentParser(description='Sentiment Analysis: War Terms and Cultural Trends')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        text_data = df.iloc[:, 0]
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2020-01-01', periods=config['data']['n_periods'], freq='D')
        sample_texts = [
            "peace and hope for the future",
            "war and conflict continue",
            "unity and progress together",
            "crisis and violence escalate"
        ]
        text_data = pd.Series(
            [np.random.choice(sample_texts) for _ in range(config['data']['n_periods'])],
            index=dates
        )
    else:
        raise ValueError("No data source specified")
    
        sentiment = calculate_sentiment_score(
        text_data,
        config['sentiment']['positive_words'],
        config['sentiment']['negative_words']
    )
    
    logging.info(f"Mean sentiment: {sentiment.mean():.4f}")
    logging.info(f"Sentiment range: [{sentiment.min():.4f}, {sentiment.max():.4f}]")
    
    plot_sentiment_trend(sentiment, "Sentiment Trend Over Time",
                        output_dir / 'sentiment_trend.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

