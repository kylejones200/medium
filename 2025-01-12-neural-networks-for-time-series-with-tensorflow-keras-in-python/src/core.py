"""Core functions for neural networks in time series forecasting."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_lagged_features(data: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged features for time series."""
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape: Tuple[int, int], units: int = 50) -> Sequential:
    """Build LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(units, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(df: pd.DataFrame, value_col: str, lag: int, train_size: float = 0.8) -> Tuple:
    """Prepare data for LSTM training."""
    scaler = MinMaxScaler()
    values = df[value_col].values.reshape(-1, 1)
    
    train_len = int(len(values) * train_size)
    train_data = values[:train_len]
    test_data = values[train_len:]
    
    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    X_train, y_train = create_lagged_features(train_scaled.flatten(), lag)
    X_test, y_test = create_lagged_features(test_scaled.flatten(), lag)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test, scaler

def plot_forecast(actual: np.ndarray, predicted: np.ndarray, title: str, output_path: Path):
 """Plot forecast vs actual """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(actual, label="Actual", color="#4A90A4", linewidth=1.2)
    ax.plot(predicted, label="Predicted", color="#D4A574", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

