"""Core functions for alternative metrics in predictive maintenance."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def calculate_health_index(df: pd.DataFrame, sensor_cols: list) -> pd.Series:
    """Calculate health index from sensor readings."""
    health = df[sensor_cols].mean(axis=1)
    return health


def calculate_degradation_rate(health: pd.Series, window: int = 10) -> pd.Series:
    """Calculate degradation rate using rolling window."""
    degradation = -health.rolling(window=window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    return degradation


def calculate_remaining_useful_life(health: pd.Series, threshold: float) -> pd.Series:
    """Estimate remaining useful life based on health index."""
    rul = pd.Series(index=health.index, dtype=float)
    for i in range(len(health)):
        remaining = health.iloc[i:]
        below_threshold = (remaining < threshold)
        if below_threshold.any():
            rul.iloc[i] = below_threshold.idxmax() - health.index[i]
        else:
            rul.iloc[i] = len(remaining)
    return rul


def plot_health_metrics(health: pd.Series, degradation: pd.Series, rul: pd.Series, 
                       threshold: float, output_path: Path):
    """Plot health metrics with Tufte style."""
    setup_tufte_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axes[0].plot(health.index, health.values, color="#4A90A4", linewidth=1.2)
    axes[0].axhline(threshold, color='red', linestyle='--', linewidth=1.2, label='Threshold')
    apply_tufte_style(axes[0], title="Health Index")
    axes[0].set_ylabel("Health Index")
    axes[0].legend(loc='best')
    
    axes[1].plot(degradation.index, degradation.values, color="#D4A574", linewidth=1.2)
    apply_tufte_style(axes[1], title="Degradation Rate")
    axes[1].set_ylabel("Degradation Rate")
    
    axes[2].plot(rul.index, rul.values, color="#8B6F9E", linewidth=1.2)
    apply_tufte_style(axes[2], title="Remaining Useful Life")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("RUL")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

