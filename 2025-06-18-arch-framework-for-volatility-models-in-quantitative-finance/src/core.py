"""Core functions for ARCH framework volatility models in quantitative finance."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from arch import arch_model
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_returns_volatility(n: int = 1000, seed: int = 42) -> Tuple[pd.Series, pd.Series]:
    """Simulate returns with volatility clustering."""
    np.random.seed(seed)
    errors = np.random.normal(size=n)
    volatility = np.zeros(n)
    returns = np.zeros(n)
    
    omega, alpha, beta = 0.1, 0.8, 0.1
    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * errors[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = volatility[t] * np.random.normal()
    
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    returns_series = pd.Series(returns, index=dates)
    volatility_series = pd.Series(volatility, index=dates)
    
    return returns_series, volatility_series

def fit_garch_model(returns: pd.Series, p: int = 1, q: int = 1):
    """Fit GARCH model to returns."""
    model = arch_model(returns * 100, vol='GARCH', p=p, q=q)
    return model.fit()

def forecast_volatility(model, horizon: int = 10) -> pd.Series:
    """Forecast volatility using fitted GARCH model."""
    forecast = model.forecast(horizon=horizon)
    return forecast.variance.iloc[-1]

def plot_volatility_analysis(returns: pd.Series, volatility: pd.Series,
                            forecast_var: pd.Series, title: str, output_path: Path):
 """Plot volatility analysis """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    axes[0].plot(returns.index, returns.values, color="#4A90A4", linewidth=1.2)
    axes[0].set_ylabel("Returns")
    
    axes[1].plot(volatility.index, volatility.values, color="#D4A574", linewidth=1.2)
    axes[1].set_ylabel("Volatility")
    
    axes[2].plot(forecast_var.values, marker='o', color="#8B6F9E", linewidth=1.2, markersize=4)
    axes[2].set_xlabel("Horizon")
    axes[2].set_ylabel("Variance")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

