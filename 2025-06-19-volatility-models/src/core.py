"""Core functions for ARCH volatility modeling."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from arch import arch_model
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def simulate_returns_with_volatility_clustering(n: int = 1000, omega: float = 0.1,
                                               alpha: float = 0.8, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate returns with volatility clustering (ARCH process)."""
    np.random.seed(seed)
    errors = np.random.normal(size=n)
    volatility = np.zeros(n)
    returns = np.zeros(n)
    
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * errors[t-1]**2)
        returns[t] = volatility[t] * np.random.normal()
    
    return returns, volatility


def fit_arch_model(returns: pd.Series, vol: str = "ARCH", p: int = 1):
    """Fit ARCH model to returns."""
    model = arch_model(returns, vol=vol, p=p)
    return model.fit()


def forecast_volatility(model, horizon: int = 10):
    """Forecast volatility using fitted ARCH model."""
    forecast = model.forecast(horizon=horizon)
    return forecast.variance.iloc[-1]


def plot_returns_volatility(returns: np.ndarray, volatility: np.ndarray, output_path: Path):
    """Plot returns and volatility with Tufte style."""
    setup_tufte_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(returns, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(axes[0], title="Simulated Returns")
    axes[0].set_ylabel("Returns")
    
    axes[1].plot(volatility, color="#D4A574", linewidth=1.2)
    apply_tufte_style(axes[1], title="Simulated Volatility")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Volatility")
    
    save_tufte_figure(output_path)


def plot_volatility_forecast(forecast_variance: pd.Series, output_path: Path):
    """Plot forecasted volatility with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(forecast_variance.values, marker="o", color="#4A90A4", 
           linewidth=1.2, markersize=4)
    apply_tufte_style(ax, title="Forecasted Volatility")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Variance")
    
    save_tufte_figure(output_path)

