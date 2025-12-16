"""Core functions for Bayesian change point detection."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_change_point_data(n: int = 100, change_point: int = 50,
                               before_mean: float = 10, after_mean: float = 20,
                               std: float = 2, seed: int = 42) -> pd.DataFrame:
    """Simulate data with a change point."""
    np.random.seed(seed)
    before = np.random.normal(before_mean, std, change_point)
    after = np.random.normal(after_mean, std, n - change_point)
    data = np.concatenate([before, after])
    
    return pd.DataFrame({
        'time': np.arange(n),
        'value': data
    })

def detect_change_point_basic(data: np.ndarray, window: int = 10) -> int:
    """Basic change point detection using sliding window."""
    n = len(data)
    max_diff = 0
    change_point = n // 2
    
    for i in range(window, n - window):
        before_mean = np.mean(data[i-window:i])
        after_mean = np.mean(data[i:i+window])
        diff = abs(after_mean - before_mean)
        
        if diff > max_diff:
            max_diff = diff
            change_point = i
    
    return change_point

def plot_change_point_detection(df: pd.DataFrame, detected_cp: int,
                                title: str, output_path: Path):
 """Plot change point detection """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['time'], df['value'], color="#4A90A4", linewidth=1.2)
    ax.axvline(detected_cp, color='red', linestyle='--', linewidth=1.5, 
              label=f'Detected Change Point: {detected_cp}')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

