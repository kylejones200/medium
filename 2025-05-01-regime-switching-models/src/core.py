"""Core functions for regime switching models."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_regime_data(n: int = 500, regime_probs: Tuple[float, float] = (0.7, 0.3),
                         stds: Tuple[float, float] = (1, 5), seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with regime switching."""
    np.random.seed(seed)
    regimes = np.random.choice([0, 1], size=n, p=regime_probs)
    data = np.array(np.random.normal(0, np.where(regimes == 0, stds[0], stds[1])))
    
    return pd.DataFrame({
        'Data': data,
        'True_Regime': regimes,
        'Time': range(n)
    })

def fit_markov_switching(data: np.ndarray, k_regimes: int = 2,
                         switching_variance: bool = True) -> Any:
    """Fit Markov switching regression model."""
    model = MarkovRegression(data, k_regimes=k_regimes, trend='c',
                            switching_variance=switching_variance)
    return model.fit()

def add_predictions(df: pd.DataFrame, result: Any) -> pd.DataFrame:
    """Add predicted probabilities and regimes to DataFrame."""
    df = df.copy()
    df['Predicted_Prob_High'] = result.smoothed_marginal_probabilities[:, 1]
    df['Predicted_Regime'] = np.argmax(result.smoothed_marginal_probabilities, axis=1)
    return df

def calculate_accuracy(df: pd.DataFrame) -> float:
    """Calculate prediction accuracy."""
    return (df['True_Regime'] == df['Predicted_Regime']).mean()

def calculate_regime_statistics(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    """Calculate statistics for each regime."""
    stats_dict = {}
    for regime in [0, 1]:
        regime_data = df[df['True_Regime'] == regime]['Data']
        stats_dict[regime] = {
            'mean': regime_data.mean(),
            'std': regime_data.std(),
            'skewness': stats.skew(regime_data),
            'kurtosis': stats.kurtosis(regime_data)
        }
    return stats_dict

def calculate_regime_durations(df: pd.DataFrame) -> Dict[int, float]:
    """Calculate average duration in each regime."""
    durations = {}
    for regime in [0, 1]:
        regime_runs = (df['Predicted_Regime'] == regime).astype(int).groupby(
            (df['Predicted_Regime'] != df['Predicted_Regime'].shift()).cumsum()
        ).sum()
        durations[regime] = regime_runs.mean()
    return durations

def plot_regime_data(df: pd.DataFrame, output_path: Path):
    """Plot original data with regime highlighting."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for regime in [0, 1]:
        mask = df['True_Regime'] == regime
        color = "#4A90A4" if regime == 0 else "#D4A574"
        ax.scatter(df[mask]['Time'], df[mask]['Data'], 
                  label=f"Regime {regime}", color=color, alpha=0.6, s=20)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_regime_comparison(df: pd.DataFrame, output_path: Path):
    """Plot true vs predicted regimes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df['Time'], df['True_Regime'], label='True Regime', 
           color="#4A90A4", linewidth=1.2, alpha=0.7)
    ax.plot(df['Time'], df['Predicted_Regime'], label='Predicted Regime', 
           color="#D4A574", linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Regime")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_density_distribution(df: pd.DataFrame, output_path: Path):
    """Plot density distribution by regime."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for regime in [0, 1]:
        color = "#4A90A4" if regime == 0 else "#D4A574"
        sns.kdeplot(data=df[df['True_Regime'] == regime]['Data'], 
                   label=f"Regime {regime}", color=color, ax=ax)
    
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_transition_matrix(result: Any, output_path: Path):
    """Plot transition probability matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    transition_matrix = result.regime_transition.reshape(2, 2)
    sns.heatmap(transition_matrix, annot=True, cmap='coolwarm', center=0.5,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(df: pd.DataFrame, output_path: Path):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    confusion_matrix = pd.crosstab(df['True_Regime'], df['Predicted_Regime'])
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_xlabel("Predicted Regime")
    ax.set_ylabel("True Regime")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

