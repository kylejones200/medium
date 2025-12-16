"""Core functions for building industrial IoT data pipeline with AWS."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_iot_data(n_points: int, n_sensors: int = 5, seed: int = 42) -> pd.DataFrame:
    """Simulate industrial IoT sensor data."""
    np.random.seed(seed)
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq='1min')
    
    data = {'timestamp': timestamps}
    for i in range(n_sensors):
        base_value = 50 + i * 10
        noise = np.random.normal(0, 5, n_points)
        trend = np.linspace(0, 10, n_points) if i % 2 == 0 else np.zeros(n_points)
        data[f'sensor_{i+1}'] = base_value + trend + noise
    
    return pd.DataFrame(data)

def analyze_sensor_data(df: pd.DataFrame, sensor_cols: list) -> Dict:
    """Analyze sensor data characteristics."""
    return {
        'n_samples': len(df),
        'n_sensors': len(sensor_cols),
        'mean_values': df[sensor_cols].mean().to_dict(),
        'std_values': df[sensor_cols].std().to_dict(),
        'missing_values': df[sensor_cols].isnull().sum().to_dict()
    }

def plot_sensor_data(df: pd.DataFrame, sensor_cols: list, title: str, output_path: Path):
 """Plot sensor data """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, col in enumerate(sensor_cols[:5]):
        ax.plot(df['timestamp'], df[col], label=col, linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Sensor Value")
    ax.legend(loc='best', ncol=2)
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

