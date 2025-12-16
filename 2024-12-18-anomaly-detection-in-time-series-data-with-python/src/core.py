"""Core functions for anomaly detection in time series data."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_lagged_features(data: pd.Series, lag: int = 5) -> pd.DataFrame:
    """Create lagged features for anomaly detection."""
    df = pd.DataFrame({'value': data})
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = data.shift(i)
    return df.dropna()

def detect_anomalies_isolation_forest(X: np.ndarray, contamination: float = 0.1,
                                     random_state: int = 42) -> np.ndarray:
    """Detect anomalies using Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(X)
    return (predictions == -1).astype(int)

def detect_anomalies_statistical(data: pd.Series, threshold: float = 3.0) -> np.ndarray:
    """Detect anomalies using statistical method (Z-score)."""
    mean = data.mean()
    std = data.std()
    z_scores = np.abs((data - mean) / std)
    return (z_scores > threshold).astype(int)

def plot_anomalies(data: pd.Series, anomalies: np.ndarray, title: str, output_path: Path):
 """Plot time series with detected anomalies """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data.index if hasattr(data.index, '__len__') else range(len(data)),
           data.values, label="Time Series", color="#4A90A4", linewidth=1.2)
    
    anomaly_indices = np.where(anomalies == 1)[0]
    if len(anomaly_indices) > 0:
        if hasattr(data.index, '__getitem__'):
            anomaly_values = data.iloc[anomaly_indices]
            ax.scatter(anomaly_values.index, anomaly_values.values,
                      color="#D4A574", s=50, label="Anomalies", zorder=5)
        else:
            ax.scatter(anomaly_indices, data.iloc[anomaly_indices].values,
                      color="#D4A574", s=50, label="Anomalies", zorder=5)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

