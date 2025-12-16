"""Core functions for matrix factorization for time series (MFLE)."""

import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_multivariate_series(n_series: int = 10, n_timesteps: int = 100,
                                time_end: float = 10, noise_std: float = 0.3,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multivariate time series data with phase shifts."""
    np.random.seed(seed)
    time = np.linspace(0, time_end, n_timesteps)
    phases = np.linspace(0, 2 * np.pi, n_series)
    data = np.array([
        np.sin(time + phase) + np.random.normal(0, noise_std, n_timesteps)
        for phase in phases
    ])
    return data, time

def apply_svd(data: np.ndarray, n_components: int = 3) -> Tuple[TruncatedSVD, np.ndarray, np.ndarray]:
    """Apply Truncated SVD for matrix factorization."""
    svd = TruncatedSVD(n_components=n_components)
    latent_features = svd.fit_transform(data)
    reconstructed = svd.inverse_transform(latent_features)
    return svd, latent_features, reconstructed

def plot_reconstruction_comparison(data: np.ndarray, reconstructed: np.ndarray,
                                  time: np.ndarray, n_series: int = 3,
                                  output_path: Path = None):
 """Plot original vs reconstructed time series """
    fig, axes = plt.subplots(n_series, 1, figsize=(12, 4 * n_series))
    
    if n_series == 1:
        axes = [axes]
    
    for i in range(n_series):
        axes[i].plot(time, data[i], label='Original', color="#4A90A4", linewidth=1.2)
        axes[i].plot(time, reconstructed[i], label='Reconstructed', 
                    color="#D4A574", linewidth=1.2, linestyle='--')
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")
        axes[i].legend(loc='best')
    
    plt.suptitle("Original vs Reconstructed Time Series (SVD)", 
                fontsize=12, y=0.98, color='0.2')
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

