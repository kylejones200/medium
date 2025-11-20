"""Core functions for ensemble time series forecasting."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure

warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=UserWarning)


def generate_synthetic_data(n_samples: int = 200, seed: int = 42) -> pd.Series:
    """Generate synthetic random walk time series."""
    np.random.seed(seed)
    return pd.Series(np.cumsum(np.random.randn(n_samples)))


def create_features(data: pd.Series) -> pd.DataFrame:
    """Create lagged features and derived variables."""
    df = pd.DataFrame({
        'value': data,
        'lag_1': data.shift(1),
        'lag_2': data.shift(2),
        'rate_of_change': data.diff()
    }).dropna()
    
    df['direction'] = (df['value'].shift(-1) > df['value']).astype(int)
    df['next_value'] = df['value'].shift(-1)
    return df.dropna()


def split_data(X: pd.DataFrame, y_class: pd.Series, y_reg: pd.Series, 
               test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Split data into training and testing sets (no shuffle for time series)."""
    return train_test_split(
        X, y_class, y_reg, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=False
    )


def fit_classifier(X_train: pd.DataFrame, y_train: pd.Series, 
                   random_state: int = 42) -> RandomForestClassifier:
    """Fit Random Forest classifier for direction prediction."""
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def fit_regressor(X_train: pd.DataFrame, y_train: pd.Series,
                  random_state: int = 42) -> RandomForestRegressor:
    """Fit Random Forest regressor for value prediction."""
    reg = RandomForestRegressor(random_state=random_state)
    reg.fit(X_train, y_train)
    return reg


def add_direction_feature(X: pd.DataFrame, clf: RandomForestClassifier) -> pd.DataFrame:
    """Add predicted direction as feature."""
    X_with_dir = X.copy()
    X_with_dir['direction_pred'] = clf.predict(X)
    return X_with_dir


def test_stationarity(series: pd.Series) -> Dict[str, float]:
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(series)
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] <= 0.05
    }


def find_best_arima(series: pd.Series, max_order: int = 3) -> Optional[Dict[str, Any]]:
    """Find best ARIMA model using grid search."""
    p = d = q = range(0, max_order)
    pdq = list(itertools.product(p, d, q))
    
    best_aic = np.inf
    best_model = None
    best_pdq = None
    
    for param in pdq:
        try:
            model = ARIMA(series, order=param)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_model = results
                best_pdq = param
        except:
            continue
    
    if best_model:
        return {
            'model': best_model,
            'order': best_pdq,
            'aic': best_aic
        }
    return None


def forecast_arima(model, steps: int, last_value: float) -> np.ndarray:
    """Generate ARIMA forecast and integrate for original scale."""
    forecast = model.get_forecast(steps=steps)
    forecast_diff = forecast.predicted_mean
    return np.cumsum(forecast_diff) + last_value


def plot_time_series(series: pd.Series, output_path: Path, title: str = "Time Series"):
    """Plot time series with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(series.index, series.values, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    
    save_tufte_figure(output_path)


def plot_predictions(actual: pd.Series, predicted: np.ndarray, output_path: Path,
                    title: str, metrics: Dict[str, float] = None):
    """Plot predictions vs actual with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(actual.values, label="Actual", color="#4A90A4", linewidth=1.2)
    ax.plot(predicted, label="Predicted", color="#D4A574", linewidth=1.2)
    
    title_text = title
    if metrics:
        if 'mae' in metrics:
            title_text += f": MAE = {metrics['mae']:.2f}"
        if 'accuracy' in metrics:
            title_text += f", Accuracy = {metrics['accuracy']:.2%}"
    
    apply_tufte_style(ax, title=title_text)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    save_tufte_figure(output_path)


def plot_residuals(residuals: np.ndarray, output_path: Path, title: str):
    """Plot residuals with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(residuals, color="#8B6F9E", linewidth=1.2)
    ax.axhline(0, color='#D4A574', linestyle='--', linewidth=1)
    
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Residual")
    
    save_tufte_figure(output_path)

