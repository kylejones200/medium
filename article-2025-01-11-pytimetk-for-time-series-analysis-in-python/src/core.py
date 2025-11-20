"""Core functions for PyTimeTK time series analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def prepare_time_series_data(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """Prepare time series data for PyTimeTK."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    return df[value_col]


def analyze_time_series_features(series: pd.Series) -> Dict:
    """Analyze time series features."""
    return {
        'length': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing',
        'volatility': series.pct_change().std()
    }


def plot_pytimetk_analysis(series: pd.Series, title: str, output_path: Path):
    """Plot PyTimeTK analysis with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(series.index, series.values, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    
    save_tufte_figure(output_path)
