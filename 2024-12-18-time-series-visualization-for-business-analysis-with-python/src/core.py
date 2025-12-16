"""Core functions for time series visualization for business analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def plot_time_series(df: pd.DataFrame, value_col: str, date_col: str, title: str, output_path: Path):
 """Plot time series """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        ax.plot(df[date_col], df[value_col], color="#4A90A4", linewidth=1.2)
        ax.set_xlabel("Date")
    else:
        ax.plot(df[value_col], color="#4A90A4", linewidth=1.2)
        ax.set_xlabel("Time")
    
    ax.set_ylabel("Value")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_multiple_series(df: pd.DataFrame, columns: List[str], title: str, output_path: Path):
 """Plot multiple time series """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ["#4A90A4", "#D4A574", "#8B6F9E", "#A8C5A0", "#E8A87C"]
    for i, col in enumerate(columns):
        ax.plot(df.index if hasattr(df.index, '__len__') else range(len(df)),
               df[col], label=col, color=colors[i % len(colors)], linewidth=1.2)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_seasonal_decomposition(df: pd.DataFrame, value_col: str, period: int, title: str, output_path: Path):
 """Plot seasonal decomposition """
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    trend = df[value_col].rolling(window=period, center=True).mean()
    seasonal = df[value_col] - trend
    residual = df[value_col] - trend - seasonal
    
    axes[0].plot(df[value_col], label="Original", color="#4A90A4", linewidth=1.2)
    axes[0].set_ylabel("Value")
    
    axes[1].plot(trend, label="Trend", color="#D4A574", linewidth=1.2)
    axes[1].set_ylabel("Trend")
    
    axes[2].plot(seasonal, label="Seasonal", color="#8B6F9E", linewidth=1.2)
    axes[2].set_ylabel("Seasonal")
    
    axes[3].plot(residual, label="Residual", color="#A8C5A0", linewidth=1.2)
    axes[3].set_xlabel("Time")
    axes[3].set_ylabel("Residual")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

