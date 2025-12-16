"""Core functions for nonparametric regression with KNN and kernel smoothing."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_temperature_data(url: str = None) -> pd.DataFrame:
    """Load temperature data from URL or local file."""
    if url is None:
        url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    return pd.read_csv(url, parse_dates=['Date'], index_col='Date')

def create_lag_features(data: pd.DataFrame, max_lag: int = 7, target_col: str = 'Temp') -> pd.DataFrame:
    """Create lag features for time series."""
    data = data.copy()
    for lag in range(1, max_lag + 1):
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    return data.dropna()

def split_time_series(data: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time series data at specified date."""
    train = data[:split_date]
    test = data[split_date:]
    return train, test

def train_knn_regressor(X_train: pd.DataFrame, y_train: pd.Series, n_neighbors: int = 5) -> KNeighborsRegressor:
    """Train k-NN regressor."""
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def apply_kernel_smoothing(data: pd.DataFrame, target_col: str = 'Temp', bw: str = 'cv_ls') -> np.ndarray:
    """Apply kernel regression smoothing."""
    data_reset = data.reset_index()
    data_reset['Time'] = np.arange(len(data_reset))
    kr = KernelReg(endog=[data_reset[target_col]], exog=[data_reset['Time']], 
                   var_type='c', bw=bw)
    y_pred, _ = kr.fit([data_reset['Time']])
    return y_pred

def plot_time_series(data: pd.DataFrame, target_col: str, output_path: Path):
 """Plot raw time series """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(data.index, data[target_col], color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_knn_predictions(y_test: pd.Series, y_pred: np.ndarray, output_path: Path):
 """Plot KNN predictions vs actual """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(y_test.index, y_test, label='Actual', color="#4A90A4", linewidth=1.2)
    ax.plot(y_test.index, y_pred, label='Predicted', color="#D4A574", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_kernel_smoothing(data: pd.DataFrame, y_pred: np.ndarray, target_col: str, output_path: Path):
 """Plot kernel smoothing results """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_reset = data.reset_index()
    ax.plot(data_reset['Date'], data_reset[target_col], label='Original', 
           alpha=0.6, color="#4A90A4", linewidth=1.2)
    ax.plot(data_reset['Date'], y_pred, label='Smoothed', 
           color="#D4A574", linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

