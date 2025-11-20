"""Core functions for Statistical Process Control (SPC) with time series."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def generate_process_data(start_date: str = "2023-01-01", periods: int = 100,
                         freq: str = "D", mean: float = 50, std: float = 2,
                         seed: int = 42) -> pd.DataFrame:
    """Generate synthetic process data with out-of-control periods."""
    np.random.seed(seed)
    time = pd.date_range(start=start_date, periods=periods, freq=freq)
    values = np.random.normal(mean, std, periods)
    
    values[30:35] += 8
    values[70:75] -= 8
    
    return pd.DataFrame({"Time": time, "Value": values})


def calculate_control_limits(df: pd.DataFrame, sigma_multiplier: float = 3.0) -> Dict[str, float]:
    """Calculate control limits (UCL, LCL, mean)."""
    mean = df["Value"].mean()
    std_dev = df["Value"].std()
    ucl = mean + sigma_multiplier * std_dev
    lcl = mean - sigma_multiplier * std_dev
    return {
        'mean': mean,
        'std_dev': std_dev,
        'ucl': ucl,
        'lcl': lcl
    }


def identify_out_of_control(df: pd.DataFrame, limits: Dict[str, float]) -> pd.Series:
    """Identify out-of-control points."""
    return (df["Value"] > limits['ucl']) | (df["Value"] < limits['lcl'])


def plot_control_chart(df: pd.DataFrame, limits: Dict[str, float],
                      out_of_control: pd.Series, output_path: Path):
    """Plot control chart with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df["Time"], df["Value"], label="Process Data", 
           color="#4A90A4", linewidth=1.2, marker="o", markersize=3)
    ax.axhline(limits['mean'], color="#8B6F9E", linestyle="--", 
              linewidth=1.2, label="Mean")
    ax.axhline(limits['ucl'], color="#D4A574", linestyle="--", 
              linewidth=1.2, label="Upper Control Limit (UCL)")
    ax.axhline(limits['lcl'], color="#D4A574", linestyle="--", 
              linewidth=1.2, label="Lower Control Limit (LCL)")
    
    if out_of_control.any():
        ax.scatter(df["Time"][out_of_control], df["Value"][out_of_control], 
                  color="#D4A574", s=50, label="Out of Control", zorder=5)
    
    apply_tufte_style(ax, title="Statistical Process Control Chart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best', ncol=2)
    
    save_tufte_figure(output_path)

