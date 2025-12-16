"""Core functions for linear algebra for fast inference."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_linear_system(n: int = 100, m: int = 50, seed: int = 42) -> tuple:
    """Create a linear system for inference."""
    np.random.seed(seed)
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true
    return A, b, x_true

def solve_linear_system(A: np.ndarray, b: np.ndarray, method: str = 'lstsq') -> np.ndarray:
    """Solve linear system using specified method."""
    if method == 'lstsq':
        return np.linalg.lstsq(A, b, rcond=None)[0]
    elif method == 'pinv':
        return np.linalg.pinv(A) @ b
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_inference_metrics(x_true: np.ndarray, x_pred: np.ndarray) -> Dict:
    """Calculate inference accuracy metrics."""
    return {
        'mse': np.mean((x_true - x_pred) ** 2),
        'mae': np.mean(np.abs(x_true - x_pred)),
        'correlation': np.corrcoef(x_true, x_pred)[0, 1]
    }

def plot_inference_results(x_true: np.ndarray, x_pred: np.ndarray, title: str, output_path: Path):
 """Plot inference results """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(x_true, x_pred, alpha=0.6, color="#4A90A4", s=30, edgecolors='none')
    min_val = min(x_true.min(), x_pred.min())
    max_val = max(x_true.max(), x_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.2,
           label='Perfect Inference')
    
    ax.set_xlabel("True Values")
    ax.set_ylabel("Inferred Values")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

