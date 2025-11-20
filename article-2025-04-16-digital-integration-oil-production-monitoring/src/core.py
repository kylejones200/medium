"""Core functions for digital integration oil production monitoring."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def analyze_production_data(df: pd.DataFrame, timestamp_col: str, 
                           production_col: str) -> pd.DataFrame:
    """Analyze oil production data."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    return df


def calculate_production_metrics(df: pd.DataFrame, production_col: str) -> Dict:
    """Calculate production performance metrics."""
    return {
        'total_production': df[production_col].sum(),
        'mean_production': df[production_col].mean(),
        'max_production': df[production_col].max(),
        'min_production': df[production_col].min(),
        'volatility': df[production_col].std() / df[production_col].mean() if df[production_col].mean() > 0 else 0,
        'trend': 'increasing' if df[production_col].iloc[-1] > df[production_col].iloc[0] else 'decreasing'
    }


def plot_production_monitoring(df: pd.DataFrame, production_col: str, 
                              title: str, output_path: Path):
    """Plot production monitoring data with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df[production_col], color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Production")
    
    save_tufte_figure(output_path)

