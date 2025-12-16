"""Core functions for scaling machine learning with Spark."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def simulate_large_dataset(n_samples: int = 10000, n_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """Simulate large dataset for Spark processing."""
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    target = (data[:, 0] > 0).astype(int)
    
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    df['label'] = target
    return df

def analyze_spark_workflow(df: pd.DataFrame) -> Dict:
    """Analyze Spark workflow characteristics."""
    return {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'spark_suitable': len(df) > 1000
    }

def plot_spark_analysis(df: pd.DataFrame, title: str, output_path: Path):
 """Plot Spark analysis """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        ax.bar(range(len(label_counts)), label_counts.values,
              color="#4A90A4", alpha=0.7, edgecolor='none')
        ax.set_xticks(range(len(label_counts)))
        ax.set_xticklabels(label_counts.index)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            ax.hist(df[numeric_cols[0]].values, bins=30, color="#4A90A4", alpha=0.7, edgecolor='none')
    
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

