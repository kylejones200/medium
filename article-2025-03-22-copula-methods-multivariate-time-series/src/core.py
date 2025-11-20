"""Core functions for copula methods in multivariate time series."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import scipy.stats as stats
from copulas.bivariate import Clayton, StudentT
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def generate_stock_interest_data(time_steps: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic stock returns and interest rates."""
    np.random.seed(seed)
    stock_returns = np.random.normal(0, 1, time_steps)
    interest_rates = 0.5 * stock_returns + np.random.normal(0, 1, time_steps)
    return pd.DataFrame({'Stock Returns': stock_returns, 'Interest Rates': interest_rates})


def generate_inflation_unemployment_data(time_steps: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic inflation and unemployment data."""
    np.random.seed(seed)
    inflation = np.random.normal(2, 1, time_steps)
    unemployment = -0.7 * inflation + np.random.normal(0, 1, time_steps)
    return pd.DataFrame({'Inflation': inflation, 'Unemployment': unemployment})


def transform_to_uniform(data: pd.Series, time_steps: int) -> np.ndarray:
    """Transform data to uniform scale using rank transformation."""
    return stats.rankdata(data) / (time_steps + 1)


def fit_clayton_copula(u: np.ndarray, v: np.ndarray) -> Clayton:
    """Fit Clayton copula to uniform marginals."""
    copula = Clayton()
    copula.fit(pd.DataFrame({'u': u, 'v': v}))
    return copula


def fit_studentt_copula(u: np.ndarray, v: np.ndarray) -> StudentT:
    """Fit Student-t copula to uniform marginals."""
    copula = StudentT()
    copula.fit(pd.DataFrame({'u': u, 'v': v}))
    return copula


def simulate_copula_forecast(copula, u_future: np.ndarray, data: pd.DataFrame,
                            var1: str, var2: str, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate forecasts using copula."""
    v_future = copula.inverse_transform(pd.DataFrame({'u': u_future}))
    
    returns_forecast = np.quantile(data[var1], u_future)
    rates_forecast = np.quantile(data[var2], v_future['v'])
    
    return returns_forecast, rates_forecast


def plot_copula_forecast(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str,
                        title: str, output_path: Path):
    """Plot copula forecast with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(x, y, alpha=0.5, s=20, color="#4A90A4", edgecolors='none')
    apply_tufte_style(ax, title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    save_tufte_figure(output_path)

