"""Core functions for Ito's Lemma and stochastic processes."""

import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def ito_log_gbm(S0: float, mu: float, sigma: float, T: float, steps: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Ito's Lemma to simulate log(S) for Geometric Brownian Motion."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=steps))
    W = np.insert(W, 0, 0)
    ln_S = np.log(S0) + (mu - 0.5 * sigma**2) * t + sigma * W
    S = np.exp(ln_S)
    return t, S, ln_S


def simulate_ou(r0: float, mu: float, theta: float, sigma: float, T: float, steps: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Ornstein-Uhlenbeck process."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    r = np.zeros(steps + 1)
    r[0] = r0
    for i in range(steps):
        dr = theta * (mu - r[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        r[i+1] = r[i] + dr
    return np.linspace(0, T, steps + 1), r


def simulate_standard_normal_from_bm(T: float, steps: int, seed: int = None) -> np.ndarray:
    """Generate standard normal from Brownian increment."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    dW = np.random.normal(0, np.sqrt(dt), size=steps)
    Z = dW / np.sqrt(dt)
    return Z


def steady_state_ou_pdf(x: np.ndarray, mu: float, theta: float, sigma: float) -> np.ndarray:
    """Compute steady-state PDF for Ornstein-Uhlenbeck process."""
    var = sigma**2 / (2 * theta)
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mu)**2 / (2 * var))


def plot_gbm_simulation(t: np.ndarray, S: np.ndarray, output_path: Path):
    """Plot GBM simulation with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, S, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title="GBM Simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    save_tufte_figure(output_path)


def plot_ou_process(t: np.ndarray, r: np.ndarray, output_path: Path):
    """Plot Ornstein-Uhlenbeck process with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, r, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title="Ornstein-Uhlenbeck Simulation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rate")
    save_tufte_figure(output_path)


def plot_standard_normal(Z: np.ndarray, output_path: Path):
    """Plot standard normal distribution from Brownian increments with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(Z, bins=50, density=True, alpha=0.6, color="#4A90A4", edgecolor='none')
    x = np.linspace(-4, 4, 200)
    ax.plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2 / 2), 
           color="#D4A574", linewidth=1.5, linestyle='--', label="PDF N(0,1)")
    apply_tufte_style(ax, title="Standard Normal from Brownian Increments")
    ax.set_xlabel("Z")
    ax.set_ylabel("Density")
    ax.legend(loc='best')
    save_tufte_figure(output_path)


def plot_ou_steady_state(x_vals: np.ndarray, pdf_vals: np.ndarray, output_path: Path):
    """Plot steady-state distribution of OU process with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals, pdf_vals, color="#4A90A4", linewidth=1.2)
    apply_tufte_style(ax, title="Steady-State Distribution of OU Process")
    ax.set_xlabel("Rate")
    ax.set_ylabel("Density")
    save_tufte_figure(output_path)

