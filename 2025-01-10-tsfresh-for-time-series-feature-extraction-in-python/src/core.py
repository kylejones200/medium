"""Core functions for TSFresh time series feature extraction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_time_series_data(n_series: int = 10, n_timesteps: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic time series data for feature extraction."""
    np.random.seed(seed)
    data = []
    for series_id in range(n_series):
        time = np.arange(n_timesteps)
        value = np.sin(time / 10) + np.random.normal(0, 0.1, n_timesteps)
        for t, v in zip(time, value):
            data.append({'id': series_id, 'time': t, 'value': v})
    return pd.DataFrame(data)

def extract_tsfresh_features(df: pd.DataFrame, column_id: str = 'id', 
                            column_sort: str = 'time', column_value: str = 'value',
                            default_fc_parameters: Optional[dict] = None) -> pd.DataFrame:
    """Extract features using TSFresh."""
    if default_fc_parameters is None:
        default_fc_parameters = ComprehensiveFCParameters()
    
    extracted_features = extract_features(
        df,
        column_id=column_id,
        column_sort=column_sort,
        column_value=column_value,
        default_fc_parameters=default_fc_parameters,
        impute_function=impute
    )
    return extracted_features

def select_relevant_features(X: pd.DataFrame, y: pd.Series, 
                            fdr_level: float = 0.05) -> pd.DataFrame:
    """Select relevant features using statistical tests."""
    selected_features = select_features(X, y, fdr_level=fdr_level)
    return selected_features

def plot_time_series(df: pd.DataFrame, output_path: Path, n_series: int = 3):
 """Plot sample time series """
    fig, axes = plt.subplots(n_series, 1, figsize=(10, 3 * n_series), sharex=True)
    
    if n_series == 1:
        axes = [axes]
    
    for i, series_id in enumerate(df['id'].unique()[:n_series]):
        series_data = df[df['id'] == series_id]
        axes[i].plot(series_data['time'], series_data['value'], 
                    color="#4A90A4", linewidth=1.2)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")
    
    plt.suptitle("Sample Time Series for Feature Extraction", 
                fontsize=12, y=0.98, color='0.2')
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

