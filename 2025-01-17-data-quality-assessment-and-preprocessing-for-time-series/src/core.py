"""Core functions for data quality assessment and preprocessing for time series."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def assess_data_quality(df: pd.DataFrame, value_col: str) -> Dict:
    """Assess data quality metrics."""
    return {
        'missing_values': df[value_col].isnull().sum(),
        'missing_percentage': (df[value_col].isnull().sum() / len(df)) * 100,
        'duplicates': df.duplicated().sum(),
        'outliers': len(df[(df[value_col] > df[value_col].quantile(0.99)) | 
                          (df[value_col] < df[value_col].quantile(0.01))]),
        'data_range': (df[value_col].max() - df[value_col].min()),
        'variance': df[value_col].var()
    }

def preprocess_time_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Preprocess time series data."""
    df = df.copy()
    
    df[value_col] = df[value_col].fillna(method='ffill').fillna(method='bfill')
    df = df.drop_duplicates()
    
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[value_col] = df[value_col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def plot_data_quality(original: pd.Series, processed: pd.Series, title: str, output_path: Path):
 """Plot data quality comparison """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.plot(original.index, original.values, color="#4A90A4", linewidth=1.2, alpha=0.7)
    ax1.set_ylabel("Value")
    
    ax2.plot(processed.index, processed.values, color="#D4A574", linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()

