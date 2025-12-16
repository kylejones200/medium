#!/usr/bin/env python3
"""
CLIP Computer Vision for Solar Panels

Main entry point for running CLIP-based solar panel analysis.
"""

import argparse
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='CLIP Computer Vision for Solar Panels')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--images-dir', type=Path, default=None, help='Directory with images')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    images_dir = Path(args.images_dir) if args.images_dir else Path(config['data']['images_dir'])
    
    if images_dir.exists() and images_dir.is_dir():
        image_paths = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        if not image_paths:
            logging.info(f"No images found in {images_dir}")
            return
    elif config['data']['generate_demo']:
                        image_paths = [Path(f"demo_image_{i}.jpg") for i in range(config['data']['n_images'])]
    else:
        raise ValueError("No images directory specified")
    
    logging.info(f"Analyzing {len(image_paths)} images...")
    results = analyze_solar_panel_images(image_paths)
    
    logging.info(f"\nDetection Results:")
    logging.info(f"Mean score: {results['detection_score'].mean():.4f}")
    logging.info(f"Max score: {results['detection_score'].max():.4f}")
    
    plot_detection_results(results, "Solar Panel Detection Results",
                          output_dir / 'detection_results.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

