"""Core functions for Value at Risk and Expected Shortfall calculation."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy.stats import norm
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_var_parametric(returns: pd.Series, confidence_level: float = 0.95,
                            time_horizon_days: int = 1) -> Dict:
    """Calculate Value at Risk using parametric method (assumes normal distribution)."""
    mean_return = returns.mean()
    std_return = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var_pct = mean_return - z_score * std_return * np.sqrt(time_horizon_days)
    return {'var_pct': var_pct, 'confidence_level': confidence_level, 'method': 'parametric'}

def calculate_var_historical(returns: pd.Series, confidence_level: float = 0.95) -> Dict:
    """Calculate Value at Risk using historical simulation."""
    var_pct = np.percentile(returns, (1 - confidence_level) * 100)
    return {'var_pct': var_pct, 'confidence_level': confidence_level, 'method': 'historical'}

def calculate_var_monte_carlo(returns: pd.Series, confidence_level: float = 0.95,
                             n_simulations: int = 10000, seed: int = 42) -> Dict:
    """Calculate Value at Risk using Monte Carlo simulation."""
    np.random.seed(seed)
    mean_return = returns.mean()
    std_return = returns.std()
    simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
    var_pct = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    return {'var_pct': var_pct, 'confidence_level': confidence_level, 'method': 'monte_carlo'}

def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Expected Shortfall (Conditional Value at Risk)."""
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    tail_losses = returns[returns <= var_threshold]
    es = tail_losses.mean() if len(tail_losses) > 0 else 0
    return es

def plot_var_comparison(var_results: Dict, output_path: Path):
 """Plot comparison of different VaR methods """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(var_results.keys())
    var_values = [var_results[m]['var_pct'] for m in methods]
    
    ax.bar(methods, var_values, color="#4A90A4", alpha=0.7, edgecolor='none')
    ax.set_ylabel("VaR (%)")
    ax.set_xlabel("Method")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

