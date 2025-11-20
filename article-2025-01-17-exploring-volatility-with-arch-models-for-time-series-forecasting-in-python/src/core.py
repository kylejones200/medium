"""Core functions for exploring volatility with ARCH models."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from arch import arch_model
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def simulate_volatility_clustering(n: int = 1000, omega: float = 0.1, alpha: float = 0.8,
                                  beta: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate returns with volatility clustering (GARCH process)."""
    np.random.seed(seed)
    errors = np.random.normal(size=n)
    volatility = np.zeros(n)
    returns = np.zeros(n)
    
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * errors[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = volatility[t] * np.random.normal()
    
    return returns, volatility


def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1):
    """Fit GARCH model to returns."""
    model = arch_model(returns, vol='GARCH', p=p, q=q)
    return model.fit()


def forecast_volatility(model, horizon: int = 10):
    """Forecast volatility using fitted GARCH model."""
    forecast = model.forecast(horizon=horizon)
    return forecast.variance.iloc[-1]


def plot_volatility_analysis(returns: np.ndarray, volatility: np.ndarray, 
                            forecast_variance: pd.Series, output_path: Path):
    """Plot volatility analysis with Tufte style."""
    setup_tufte_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axes[0].plot(returns, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(axes[0], title="Simulated Returns")
    axes[0].set_ylabel("Returns")
    
    axes[1].plot(volatility, color="#D4A574", linewidth=1.2)
    apply_tufte_style(axes[1], title="Simulated Volatility")
    axes[1].set_ylabel("Volatility")
    
    axes[2].plot(forecast_variance.values, marker="o", color="#8B6F9E", linewidth=1.2, markersize=4)
    apply_tufte_style(axes[2], title="Forecasted Volatility")
    axes[2].set_xlabel("Horizon")
    axes[2].set_ylabel("Variance")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

