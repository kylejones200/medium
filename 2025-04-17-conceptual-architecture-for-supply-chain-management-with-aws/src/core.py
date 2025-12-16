"""Core functions for conceptual supply chain management architecture with AWS."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_supply_chain_data(n_points: int, n_nodes: int = 4, seed: int = 42) -> pd.DataFrame:
    """Simulate supply chain data."""
    np.random.seed(seed)
    timestamps = pd.date_range('2023-01-01', periods=n_points, freq='1H')
    
    data = {'timestamp': timestamps}
    nodes = ['supplier', 'manufacturer', 'distributor', 'retailer']
    for i, node in enumerate(nodes[:n_nodes]):
        base_value = 1000 - i * 100
        noise = np.random.normal(0, 50, n_points)
        trend = np.cumsum(np.random.normal(0, 2, n_points))
        data[f'{node}_inventory'] = base_value + trend + noise
    
    return pd.DataFrame(data)

def analyze_supply_chain(df: pd.DataFrame, inventory_cols: list) -> Dict:
    """Analyze supply chain data."""
    return {
        'n_samples': len(df),
        'n_nodes': len(inventory_cols),
        'mean_inventory': df[inventory_cols].mean().to_dict(),
        'total_inventory': df[inventory_cols].sum(axis=1).mean()
    }

def plot_supply_chain(df: pd.DataFrame, inventory_cols: list, title: str, output_path: Path):
 """Plot supply chain data """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for col in inventory_cols:
        ax.plot(df['timestamp'], df[col], label=col.replace('_', ' ').title(), 
               linewidth=1.2, alpha=0.7)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Inventory")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

