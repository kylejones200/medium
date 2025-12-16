"""Core functions for Fourier Transform time series analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from scipy.fft import fft, ifft, fftfreq
from sktime.datasets import load_airline
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_synthetic_signal(time: np.ndarray, freq1: float, freq2: float, 
                             amplitude2: float = 0.5) -> np.ndarray:
    """Generate synthetic signal with two frequencies."""
    return np.sin(2 * np.pi * freq1 * time) + amplitude2 * np.sin(2 * np.pi * freq2 * time)

def compute_fft(signal: np.ndarray, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT and return frequencies and FFT result."""
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_result), d=sample_rate)
    return frequencies, fft_result

def filter_noise(fft_result: np.ndarray, frequencies: np.ndarray, 
                threshold: float) -> np.ndarray:
    """Filter frequencies above threshold."""
    fft_filtered = fft_result.copy()
    fft_filtered[np.abs(frequencies) > threshold] = 0
    return fft_filtered

def inverse_fft(fft_filtered: np.ndarray) -> np.ndarray:
    """Inverse FFT to reconstruct signal."""
    return np.fft.ifft(fft_filtered).real

def load_airline_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load airline passenger dataset."""
    y = load_airline()
    y = y.values
    time = np.arange(len(y))
    return time, y

def add_noise(signal: np.ndarray, noise_level: float = 50, seed: int = None) -> np.ndarray:
    """Add Gaussian noise to signal."""
    if seed is not None:
        np.random.seed(seed)
    return signal + np.random.normal(0, noise_level, len(signal))

def plot_time_series(time: np.ndarray, signal: np.ndarray, output_path: Path,
                    title: str, xlabel: str = "Time", ylabel: str = "Amplitude"):
 """Plot time series """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(time, signal, color="#4A90A4", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_frequency_domain(frequencies: np.ndarray, fft_result: np.ndarray,
                         output_path: Path, title: str):
 """Plot frequency domain representation """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    n = len(frequencies) // 2
    ax.plot(frequencies[:n], np.abs(fft_result)[:n], color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_noise_filtering(time: np.ndarray, noisy_signal: np.ndarray,
                         filtered_signal: np.ndarray, output_path: Path):
 """Plot noise filtering comparison """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time, noisy_signal, label="Noisy Signal", color="#8B6F9E", 
           linewidth=1.2, alpha=0.5)
    ax.plot(time, filtered_signal, label="Filtered Signal", color="#D4A574", linewidth=1.2)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

