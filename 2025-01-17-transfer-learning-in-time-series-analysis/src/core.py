"""Core functions for transfer learning in time series."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_features(data: pd.Series, lag: int = 5) -> pd.DataFrame:
    """Create lagged features for time series."""
    df = pd.DataFrame({'target': data})
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = data.shift(i)
    return df.dropna()

def train_source_model(X_source: np.ndarray, y_source: np.ndarray) -> RandomForestRegressor:
    """Train model on source domain."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_source, y_source)
    return model

def fine_tune_model(model: RandomForestRegressor, X_target: np.ndarray, y_target: np.ndarray) -> RandomForestRegressor:
    """Fine-tune model on target domain."""
    model.fit(X_target, y_target)
    return model

def plot_transfer_learning(y_source: np.ndarray, y_target: np.ndarray,
                          pred_source: np.ndarray, pred_target: np.ndarray,
                          output_path: Path):
 """Plot transfer learning results """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(y_source, label="Actual", color="#4A90A4", linewidth=1.2)
    axes[0].plot(pred_source, label="Predicted", color="#D4A574", linewidth=1.2)
    axes[0].legend(loc='best')
    
    axes[1].plot(y_target, label="Actual", color="#4A90A4", linewidth=1.2)
    axes[1].plot(pred_target, label="Predicted", color="#D4A574", linewidth=1.2)
    axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

