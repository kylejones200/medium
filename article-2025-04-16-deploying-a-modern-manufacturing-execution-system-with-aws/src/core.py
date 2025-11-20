"""Core functions for deploying manufacturing execution system with AWS."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def analyze_manufacturing_data(df: pd.DataFrame, timestamp_col: str, 
                               production_col: str) -> pd.DataFrame:
    """Analyze manufacturing production data."""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    return df


def calculate_production_metrics(df: pd.DataFrame, production_col: str) -> Dict:
    """Calculate manufacturing production metrics."""
    return {
        'total_production': df[production_col].sum(),
        'mean_production': df[production_col].mean(),
        'std_production': df[production_col].std(),
        'efficiency': df[production_col].mean() / df[production_col].max() if df[production_col].max() > 0 else 0
    }


def plot_production_trend(df: pd.DataFrame, production_col: str, title: str, output_path: Path):
    """Plot production trend with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df.index, df[production_col], color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Production")
    
    save_tufte_figure(output_path)

