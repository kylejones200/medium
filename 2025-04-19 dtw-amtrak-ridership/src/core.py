"""Core functions for DTW (Dynamic Time Warping) analysis of Amtrak ridership."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def prepare_time_series_matrix(df: pd.DataFrame, station_col: str, time_col: str,
                               value_col: str) -> pd.DataFrame:
    """Prepare time series matrix for DTW analysis."""
    pivot_df = df.pivot_table(index=station_col, columns=time_col, values=value_col, fill_value=0)
    return pivot_df


def compute_dtw_distance_matrix(ts_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute DTW distance matrix for time series."""
    scaler = TimeSeriesScalerMeanVariance()
    series_array = ts_matrix.to_numpy().reshape((ts_matrix.shape[0], ts_matrix.shape[1], 1))
    series_scaled = scaler.fit_transform(series_array)
    
    dtw_matrix = cdist_dtw(series_scaled)
    dtw_df = pd.DataFrame(dtw_matrix, index=ts_matrix.index, columns=ts_matrix.index)
    return dtw_df


def find_similar_stations(dtw_df: pd.DataFrame, target_station: str, n: int = 5) -> pd.Series:
    """Find n most similar stations to target station."""
    if target_station not in dtw_df.index:
        raise ValueError(f"Station {target_station} not found in data")
    
    similar = dtw_df.loc[target_station].sort_values().iloc[1:n+1]
    return similar


def plot_dtw_matrix(dtw_df: pd.DataFrame, title: str, output_path: Path):
    """Plot DTW distance matrix with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(dtw_df.values, cmap='viridis', aspect='auto', origin='lower')
    
    if len(dtw_df) <= 20:
        ax.set_xticks(range(len(dtw_df)))
        ax.set_yticks(range(len(dtw_df)))
        ax.set_xticklabels(dtw_df.columns, rotation=45, ha='right')
        ax.set_yticklabels(dtw_df.index)
    
    plt.colorbar(im, ax=ax, label='DTW Distance')
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Station")
    ax.set_ylabel("Station")
    
    save_tufte_figure(output_path)

