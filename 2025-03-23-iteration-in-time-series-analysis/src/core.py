"""Core functions for iteration in time series analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def iterative_forecast(data: pd.Series, n_iterations: int = 5, horizon: int = 10) -> List[np.ndarray]:
    """Perform iterative forecasting with refinement."""
    forecasts = []
    current_data = data.copy()
    
    for i in range(n_iterations):
        forecast = current_data[-horizon:].values + np.random.normal(0, 0.1, horizon)
        forecasts.append(forecast)
        current_data = pd.concat([current_data, pd.Series(forecast)])
    
    return forecasts

def calculate_iteration_metrics(forecasts: List[np.ndarray], actual: np.ndarray) -> Dict:
    """Calculate metrics for each iteration."""
    metrics = []
    for i, forecast in enumerate(forecasts):
        if len(forecast) <= len(actual):
            mse = mean_squared_error(actual[:len(forecast)], forecast)
            metrics.append({'iteration': i+1, 'mse': mse})
    return metrics

def plot_iteration_results(data: pd.Series, forecasts: List[np.ndarray], title: str, output_path: Path):
 """Plot iteration results """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data.values, label="Historical Data", color="#4A90A4", linewidth=1.2)
    
    colors = ["#D4A574", "#8B6F9E", "#A8C5A0", "#E8A87C", "#95B8D1"]
    for i, forecast in enumerate(forecasts):
        start_idx = len(data)
        x_range = range(start_idx, start_idx + len(forecast))
        ax.plot(x_range, forecast, label=f"Iteration {i+1}", 
               color=colors[i % len(colors)], linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

