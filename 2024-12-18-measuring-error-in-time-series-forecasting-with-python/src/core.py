"""Core functions for measuring error in time series forecasting."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate comprehensive error metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mean_error': np.mean(y_true - y_pred),
        'std_error': np.std(y_true - y_pred)
    }

def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path):
 """Plot error analysis """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(y_true, label="Actual", color="#4A90A4", linewidth=1.2)
    axes[0].plot(y_pred, label="Predicted", color="#D4A574", linewidth=1.2)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc='best')
    
    errors = y_true - y_pred
    axes[1].plot(errors, color="#8B6F9E", linewidth=1.2)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

