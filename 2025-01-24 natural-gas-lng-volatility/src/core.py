"""Core functions for natural gas and LNG volatility analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std()

def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for price series."""
    return df.corr()

def analyze_price_relationships(df: pd.DataFrame, price_cols: list) -> Dict:
    """Analyze relationships between price series."""
    correlations = df[price_cols].corr()
    
    returns = df[price_cols].pct_change().dropna()
    volatilities = {col: returns[col].std() for col in price_cols}
    
    return {
        'correlations': correlations,
        'volatilities': volatilities,
        'mean_prices': df[price_cols].mean().to_dict()
    }

def plot_volatility_analysis(df: pd.DataFrame, price_cols: list, title: str, output_path: Path):
 """Plot volatility analysis """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    returns = df[price_cols].pct_change().dropna()
    for col in price_cols:
        vol = calculate_volatility(returns[col])
        ax.plot(vol.index, vol.values, label=col, linewidth=1.2, alpha=0.8)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Volatility")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

