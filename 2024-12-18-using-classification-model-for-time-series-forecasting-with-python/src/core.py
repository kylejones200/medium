"""Core functions for using classification models for time series forecasting."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_classification_targets(data: pd.Series, n_bins: int = 3) -> Tuple[np.ndarray, LabelEncoder]:
    """Create classification targets by binning continuous values."""
    labels = pd.qcut(data, q=n_bins, labels=False, duplicates='drop')
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder

def create_lagged_features(data: pd.Series, lag: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged features for classification."""
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i].values)
        y.append(data[i])
    return np.array(X), np.array(y)

def train_classification_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest classifier for time series."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def plot_classification_results(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path):
 """Plot classification results """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = np.arange(len(y_true))
    ax.plot(time, y_true, label="Actual Class", color="#4A90A4", linewidth=1.2, marker='o', markersize=3)
    ax.plot(time, y_pred, label="Predicted Class", color="#D4A574", linewidth=1.2, marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Class")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

