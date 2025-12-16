"""Core functions for data visualization."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_sales_data(months: int = 12, regions: list = None, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sales data."""
    if regions is None:
        regions = ["North", "South", "West"]
    
    np.random.seed(seed)
    date_range = pd.date_range("2022-01-01", periods=months, freq="M")
    data = {
        "Month": np.tile(date_range, len(regions)),
        "Region": np.repeat(regions, len(date_range)),
        "Sales": np.random.randint(90, 150, size=len(date_range) * len(regions))
    }
    return pd.DataFrame(data)

def plot_seaborn_line(df: pd.DataFrame, x: str, y: str, hue: str, output_path: Path):
    """Plot time series with seaborn."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_weekly_trend(df: pd.DataFrame, output_path: Path):
 """Plot weekly sales trend """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weekly = df.groupby('Week')['Sales'].sum().reset_index()
    ax.plot(weekly['Week'], weekly['Sales'], marker='o', color="#4A90A4", linewidth=1.2)
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Sales')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_store_comparison(df: pd.DataFrame, output_path: Path):
 """Plot store-wise comparison """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot(index='Week', columns='Store', values='Sales').reset_index()
    colors = ["#4A90A4", "#D4A574", "#8B6F9E", "#A8C5A0"]
    for i, store in enumerate(pivot.columns[1:]):
        ax.plot(pivot['Week'], pivot[store], label=store, 
               color=colors[i % len(colors)], linewidth=1.2, marker='o')
    
    ax.set_xlabel('Week')
    ax.set_ylabel('Sales')
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

