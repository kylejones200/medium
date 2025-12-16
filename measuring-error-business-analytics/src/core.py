"""Core functions for measuring error in business analytics."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
        'r2': r2_score(y_true, y_pred),
        'mean_error': np.mean(y_true - y_pred),
        'std_error': np.std(y_true - y_pred)
    }

def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, errors: np.ndarray,
                       title: str, output_path: Path):
 """Plot error analysis """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    axes[0].scatter(y_true, y_pred, alpha=0.6, color="#4A90A4", s=30, edgecolors='none')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.2,
                label='Perfect Prediction')
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].legend(loc='best')
    
    axes[1].plot(errors, color="#D4A574", linewidth=1.2)
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Error")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

