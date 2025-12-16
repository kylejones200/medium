"""Core functions for portfolio management."""

import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_asset_returns(n_assets: int = 3, n_periods: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic asset return data."""
    np.random.seed(seed)
    mean_returns = np.random.uniform(0.05, 0.15, n_assets)
    cov_matrix = np.random.uniform(0.001, 0.02, size=(n_assets, n_assets))
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    np.fill_diagonal(cov_matrix, np.random.uniform(0.01, 0.03, n_assets))
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_periods)
    return returns, mean_returns, cov_matrix

def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> Tuple[float, float]:
    """Compute portfolio return and risk."""
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, n_portfolios: int = 1000, seed: int = None) -> Tuple[np.ndarray, list]:
    """Generate efficient frontier."""
    if seed is not None:
        np.random.seed(seed)
    n_assets = len(mean_returns)
    results = np.zeros((n_portfolios, 3))
    weights_record = []
    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = ret / vol
        results[i, 0], results[i, 1], results[i, 2] = ret, vol, sharpe
        weights_record.append(weights)
    return results, weights_record

def calculate_alpha_beta(asset_returns: np.ndarray, market_returns: np.ndarray, risk_free: float = 0.0) -> Tuple[float, float]:
    """Calculate alpha and beta."""
    excess_asset = asset_returns - risk_free
    excess_market = market_returns - risk_free
    cov = np.cov(excess_asset, excess_market)
    beta = cov[0, 1] / cov[1, 1]
    alpha = np.mean(excess_asset) - beta * np.mean(excess_market)
    return alpha, beta

def plot_efficient_frontier(results: np.ndarray, output_path: Path):
 """Plot efficient frontier """
    fig, ax = plt.subplots(figsize=(10, 4))
    scatter = ax.scatter(results[:, 1], results[:, 0], c=results[:, 2], 
                        cmap='viridis', alpha=0.6, s=20, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_alpha_beta(market_returns: np.ndarray, asset_returns: np.ndarray,
                   alpha: float, beta: float, output_path: Path):
 """Plot alpha and beta estimation """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(market_returns, asset_returns, alpha=0.4, s=20, 
              color="#4A90A4", edgecolors='none')
    ax.set_xlabel("Market Returns")
    ax.set_ylabel("Asset Returns")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

