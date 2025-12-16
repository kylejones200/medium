"""Core functions for basic data analysis using Pandas."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_and_explore_data(file_path: Path) -> pd.DataFrame:
    """Load and perform basic exploration of data."""
    df = pd.read_csv(file_path)
    return df

def calculate_summary_statistics(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics for numerical columns."""
    return {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'describe': df.describe(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }

def plot_data_distribution(df: pd.DataFrame, column: str, title: str, output_path: Path):
 """Plot data distribution """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if df[column].dtype in ['int64', 'float64']:
        ax.hist(df[column].dropna(), bins=30, color="#4A90A4", alpha=0.7, edgecolor='none')
    else:
        value_counts = df[column].value_counts().head(10)
        ax.bar(range(len(value_counts)), value_counts.values, color="#4A90A4", alpha=0.7, edgecolor='none')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

