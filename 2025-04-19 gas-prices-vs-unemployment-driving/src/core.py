"""Core functions for gas prices vs unemployment driving analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def prepare_economic_data(df: pd.DataFrame, gas_col: str, unemployment_col: str) -> Tuple:
    """Prepare data for economic analysis."""
    gas = df[gas_col].values
    unemployment = df[unemployment_col].values
    return gas, unemployment


def analyze_correlation(gas: np.ndarray, unemployment: np.ndarray) -> dict:
    """Analyze correlation between gas prices and unemployment."""
    correlation = np.corrcoef(gas, unemployment)[0, 1]
    
    X = gas.reshape(-1, 1)
    y = unemployment
    model = LinearRegression()
    model.fit(X, y)
    r2 = r2_score(y, model.predict(X))
    
    return {
        'correlation': correlation,
        'r2': r2,
        'slope': model.coef_[0],
        'intercept': model.intercept_
    }


def plot_economic_relationship(gas: np.ndarray, unemployment: np.ndarray,
                              title: str, output_path: Path):
    """Plot economic relationship with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(gas, unemployment, alpha=0.6, color="#4A90A4", s=30, edgecolors='none')
    
    X = gas.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, unemployment)
    y_pred = model.predict(X)
    ax.plot(gas, y_pred, 'r-', linewidth=1.2, label='Trend Line')
    
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Gas Prices")
    ax.set_ylabel("Unemployment Rate")
    ax.legend(loc='best')
    
    save_tufte_figure(output_path)

