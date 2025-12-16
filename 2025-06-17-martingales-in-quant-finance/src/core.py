"""Core functions for martingales in quantitative finance."""

import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_binomial_tree(S0: float, u: float, d: float, r: float, steps: int, seed: int = None) -> list:
    """Simulate binomial tree with risk-neutral probabilities."""
    if seed is not None:
        np.random.seed(seed)
    q = (1 + r - d) / (u - d)
    paths = [S0]
    for _ in range(steps):
        move = np.random.choice([u, d], p=[q, 1 - q])
        paths.append(paths[-1] * move)
    return paths

def simulate_exponential_martingale(theta: float, T: float, steps: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate exponential martingale from Brownian motion."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=steps))
    W = np.insert(W, 0, 0)
    t = np.linspace(0, T, steps + 1)
    Z = np.exp(-theta * W - 0.5 * theta**2 * t)
    return t, W, Z

def simulate_girsanov(theta: float, T: float, steps: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Girsanov change of measure (original and shifted Brownian motion)."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=steps))
    W = np.insert(W, 0, 0)
    t = np.linspace(0, T, steps + 1)
    W_tilde = W + theta * t
    return t, W, W_tilde

def plot_binomial_path(path: list, output_path: Path):
 """Plot binomial tree path """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(path, color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_exponential_martingale(t: np.ndarray, Z: np.ndarray, output_path: Path):
 """Plot exponential martingale """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, Z, color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Z(t)")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_girsanov_transformation(t: np.ndarray, W: np.ndarray, W_tilde: np.ndarray, output_path: Path):
 """Plot Girsanov transformation """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, W, label="Original Brownian Motion", color="#4A90A4", linewidth=1.2)
    ax.plot(t, W_tilde, label="Shifted (Girsanov)", color="#D4A574", linewidth=1.2, linestyle='--')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

