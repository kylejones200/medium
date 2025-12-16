"""Core functions for ensemble models with ordered time series."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_ordered_features(data: pd.Series, lag: int = 5) -> pd.DataFrame:
    """Create ordered features preserving time sequence."""
    df = pd.DataFrame({'target': data})
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = data.shift(i)
    return df.dropna()

def train_ensemble_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train multiple models for ensemble."""
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X, y)
        trained[name] = model
    
    return trained

def ensemble_predict(models: Dict, X: np.ndarray, method: str = 'mean') -> np.ndarray:
    """Generate ensemble predictions."""
    predictions = []
    for model in models.values():
        predictions.append(model.predict(X))
    
    predictions = np.array(predictions)
    
    if method == 'mean':
        return predictions.mean(axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    else:
        return predictions.mean(axis=0)

def plot_ensemble_forecast(actual: np.ndarray, individual: Dict[str, np.ndarray],
                          ensemble: np.ndarray, title: str, output_path: Path):
 """Plot ensemble forecast """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(actual, label="Actual", color="#4A90A4", linewidth=1.2)
    
    colors = ["#D4A574", "#8B6F9E"]
    for i, (name, pred) in enumerate(individual.items()):
        ax.plot(pred, label=name, color=colors[i % len(colors)], linewidth=1.2, alpha=0.7)
    
    ax.plot(ensemble, label="Ensemble", color="#A8C5A0", linewidth=1.5, linestyle='--')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

