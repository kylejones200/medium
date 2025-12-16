"""Core functions for asset price modeling."""

import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_wiener_process(n_steps: int = 1000, T: float = 1.0, seed: int = None) -> np.ndarray:
    """Simulate a Wiener process (Brownian motion)."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    increments = np.random.normal(0, np.sqrt(dt), size=n_steps)
    W = np.cumsum(increments)
    W = np.insert(W, 0, 0)
    return W

def simulate_gbm(S0: float = 100, mu: float = 0.05, sigma: float = 0.2, 
                 T: float = 1.0, n_steps: int = 1000, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate geometric Brownian motion."""
    W = simulate_wiener_process(n_steps, T, seed)
    t = np.linspace(0, T, n_steps + 1)
    exponent = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(exponent)
    return t, S

def plot_wiener_process(W: np.ndarray, t: np.ndarray, output_path: Path):
 """Plot Wiener process """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, W, color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("W(t)")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_gbm_simulation(t: np.ndarray, S: np.ndarray, output_path: Path):
 """Plot geometric Brownian motion """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, S, color="#4A90A4", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("S(t)")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

