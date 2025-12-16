#!/usr/bin/env python3
"""
Fourier Transformations for Time Series Analysis

Main entry point for running Fourier transform analysis.
"""

import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from src.core import ((level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_synthetic_signal,
    compute_fft,
    filter_noise,
    inverse_fft,
    load_airline_data,
    add_noise,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Fourier Transformations for Time Series')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if config['analysis']['run_synthetic']:
                np.random.seed(config['data']['synthetic']['seed'])
        time = np.linspace(0, config['data']['synthetic']['time_end'], 
                         config['data']['synthetic']['n_samples'])
        signal = generate_synthetic_signal(
            time,
            config['data']['synthetic']['freq1'],
            config['data']['synthetic']['freq2'],
            config['data']['synthetic']['amplitude2']
        )
        
        plot_time_series(time, signal, output_dir / 'synthetic_signal.png',
                        "Time Series: Combination of Two Sine Waves")
        
        sample_rate = time[1] - time[0]
        frequencies, fft_result = compute_fft(signal, sample_rate)
        plot_frequency_domain(frequencies, fft_result,
                            output_dir / 'frequency_domain.png',
                            "Frequency Domain Representation")
    
    if config['analysis']['run_airline']:
                time, y = load_airline_data()
        
        plot_time_series(time, y, output_dir / 'airline_data.png',
                       "Original Airline Passenger Data", 
                       xlabel="Time", ylabel="Passengers")
        
        frequencies, fft_result = compute_fft(y, sample_rate=1)
        plot_frequency_domain(frequencies, fft_result,
                            output_dir / 'airline_frequency_domain.png',
                            "Frequency Domain of Airline Passenger Data")
    
    if config['analysis']['run_noise_filtering']:
                time, y = load_airline_data()
        noisy_signal = add_noise(y, config['data']['noise']['level'], 
                               config['data']['synthetic']['seed'])
        
        frequencies, fft_result_noisy = compute_fft(noisy_signal, sample_rate=1)
        fft_filtered = filter_noise(fft_result_noisy, frequencies, 
                                  config['data']['noise']['threshold'])
        filtered_signal = inverse_fft(fft_filtered)
        
        plot_noise_filtering(time, noisy_signal, filtered_signal,
                           output_dir / 'noise_filtering.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

