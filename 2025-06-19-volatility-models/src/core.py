"""Core functions for ARCH volatility modeling."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from arch import arch_model
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

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
 """Plot returns and volatility """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    axes[0].plot(returns, color="#4A90A4", linewidth=1.2)
    axes[0].set_ylabel("Returns")
    
    axes[1].plot(volatility, color="#D4A574", linewidth=1.2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Volatility")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_volatility_forecast(forecast_variance: pd.Series, output_path: Path):
 """Plot forecasted volatility """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(forecast_variance.values, marker="o", color="#4A90A4", 
           linewidth=1.2, markersize=4)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Variance")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

