"""Core functions for time series clustering with aeon."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from aeon.datasets import make_example_3_class_dataset
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_dataset(n_instances: int = 50, n_timepoints: int = 30, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic time series dataset."""
    return make_example_3_class_dataset(n_instances=n_instances, 
                                       n_timepoints=n_timepoints, 
                                       random_state=random_state)

def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
              random_state: int = 42) -> Tuple:
    """Split data into training and testing sets (no shuffle for time series)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

def fit_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                  n_neighbors: int = 1) -> KNeighborsTimeSeriesClassifier:
    """Fit K-Neighbors time series classifier."""
    clf = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    return clf

def evaluate_classifier(clf: KNeighborsTimeSeriesClassifier, X_test: np.ndarray, 
                       y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate classifier and return metrics."""
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {
        'accuracy': accuracy,
        'predictions': y_pred
    }

def plot_sample_series(X: np.ndarray, labels: list, output_path: Path):
 """Plot sample series from each class """
    plot_series(X[0], X[1], X[2], labels=labels)
    plt.title("Sample Series from Each Class", pad=10)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

