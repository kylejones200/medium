"""Core functions for linear regression for business analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def prepare_regression_data(df: pd.DataFrame, feature_cols: list, target_col: str) -> Tuple:
    """Prepare data for linear regression."""
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y

def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, dict]:
    """Fit linear regression model and calculate metrics."""
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    metrics = {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'coefficients': model.coef_.tolist(),
        'intercept': model.intercept_
    }
    
    return model, metrics

def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray,
                           title: str, output_path: Path):
 """Plot regression results """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.6, color="#4A90A4", s=30, edgecolors='none')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.2, label='Perfect Prediction')
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

