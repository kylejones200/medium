"""Core functions for state space models and Kalman filtering."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from scipy import linalg
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def kalman_filter(y: np.ndarray, F: np.ndarray, H: np.ndarray, Q: np.ndarray,
                  R: np.ndarray, x0: np.ndarray, P0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kalman filter for state space model."""
    n = len(y)
    x = np.zeros((n, len(x0)))
    P = np.zeros((n, len(x0), len(x0)))
    
    x[0] = x0
    P[0] = P0
    
    for t in range(1, n):
        x_pred = F @ x[t-1]
        P_pred = F @ P[t-1] @ F.T + Q
        
        y_pred = H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ linalg.inv(S)
        
        x[t] = x_pred + K @ (y[t] - y_pred)
        P[t] = (np.eye(len(x0)) - K @ H) @ P_pred
    
    return x, P

def plot_kalman_filter(y: np.ndarray, x_filtered: np.ndarray, title: str, output_path: Path):
 """Plot Kalman filter results """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(y, label="Observed", color="#4A90A4", linewidth=1.2, alpha=0.7)
    ax.plot(x_filtered[:, 0], label="Filtered State", color="#D4A574", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

