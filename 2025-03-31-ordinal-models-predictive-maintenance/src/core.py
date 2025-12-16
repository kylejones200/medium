"""Core functions for ordinal models in predictive maintenance."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_ordinal_targets(data: pd.Series, n_levels: int = 4) -> Tuple[np.ndarray, LabelEncoder]:
    """Create ordinal targets for degradation levels."""
    labels = pd.qcut(data, q=n_levels, labels=False, duplicates='drop')
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder

def create_maintenance_features(df: pd.DataFrame, sensor_cols: list) -> np.ndarray:
    """Create features for maintenance prediction."""
    features = []
    for col in sensor_cols:
        features.append(df[col].values)
    return np.column_stack(features)

def train_ordinal_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train ordinal classification model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def plot_ordinal_predictions(actual: np.ndarray, predicted: np.ndarray,
                            levels: list, title: str, output_path: Path):
 """Plot ordinal predictions """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time = np.arange(len(actual))
    ax.plot(time, actual, label="Actual Level", color="#4A90A4", linewidth=1.2, marker='o', markersize=4)
    ax.plot(time, predicted, label="Predicted Level", color="#D4A574", linewidth=1.2, 
           marker='s', markersize=4, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Degradation Level")
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels)
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
