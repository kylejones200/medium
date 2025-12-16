"""Core functions for working with missing data in business analytics."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_data_with_missing(url: str = None, file_path: Path = None) -> pd.DataFrame:
    """Load data and create artificial missingness for demonstration."""
    if file_path and Path(file_path).exists():
        df = pd.read_csv(file_path)
    elif url:
        df = pd.read_csv(url)
    else:
        dates = pd.date_range('2020-01-01', periods=144, freq='M')
        passengers = 100 + 20 * np.sin(np.arange(144) / 12) + np.random.normal(0, 10, 144)
        df = pd.DataFrame({'Month': dates, 'Passengers': passengers})
    
    if 'Month' in df.columns and 'Passengers' in df.columns:
        df.loc[10:15, 'Passengers'] = np.nan
        df['Revenue'] = df['Passengers'] * 10 + np.random.normal(0, 50, size=len(df))
        df.loc[5:10, 'Revenue'] = np.nan
    
    return df

def analyze_missing_data(df: pd.DataFrame) -> Dict:
    """Analyze missing data patterns."""
    return {
        'missing_counts': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'total_missing': df.isnull().sum().sum()
    }

def impute_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """Impute missing values using specified strategy."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    elif strategy == 'iterative':
        imputer = IterativeImputer(random_state=42)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def plot_missing_analysis(df: pd.DataFrame, title: str, output_path: Path):
 """Plot missing data analysis """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        ax.bar(range(len(missing_counts)), missing_counts.values,
              color="#4A90A4", alpha=0.7, edgecolor='none')
        ax.set_xticks(range(len(missing_counts)))
        ax.set_xticklabels(missing_counts.index, rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No missing values', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

