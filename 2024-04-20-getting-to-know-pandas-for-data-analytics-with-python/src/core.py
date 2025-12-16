"""Core functions for getting to know Pandas for data analytics."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def perform_data_operations(df: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
    """Perform common Pandas data operations."""
    result = df.copy()
    
    for op in operations:
        if op == 'groupby':
            if 'category' in result.columns:
                result = result.groupby('category').agg({
                    col: 'mean' for col in result.select_dtypes(include=[np.number]).columns
                })
        elif op == 'sort':
            if len(result.select_dtypes(include=[np.number]).columns) > 0:
                result = result.sort_values(by=result.select_dtypes(include=[np.number]).columns[0], ascending=False)
        elif op == 'filter':
            if len(result.select_dtypes(include=[np.number]).columns) > 0:
                result = result[result[result.select_dtypes(include=[np.number]).columns[0]] > result[result.select_dtypes(include=[np.number]).columns[0]].median()]
    
    return result

def analyze_dataframe(df: pd.DataFrame) -> Dict:
    """Analyze dataframe structure and content."""
    return {
        'info': df.info(),
        'head': df.head(),
        'tail': df.tail(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().to_dict()
    }

def plot_dataframe_comparison(df1: pd.DataFrame, df2: pd.DataFrame,
                              column: str, title: str, output_path: Path):
 """Plot comparison between two dataframes """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if column in df1.columns and column in df2.columns:
        if df1[column].dtype in ['int64', 'float64']:
            ax.hist(df1[column].dropna(), bins=30, alpha=0.6, label="Original", 
                   color="#4A90A4", edgecolor='none')
            ax.hist(df2[column].dropna(), bins=30, alpha=0.6, label="Processed", 
                   color="#D4A574", edgecolor='none')
        else:
            counts1 = df1[column].value_counts().head(10)
            counts2 = df2[column].value_counts().head(10)
            x = np.arange(len(counts1))
            ax.bar(x - 0.2, counts1.values, 0.4, label="Original", color="#4A90A4", alpha=0.7)
            ax.bar(x + 0.2, counts2.values, 0.4, label="Processed", color="#D4A574", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(counts1.index, rotation=45, ha='right')
    
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

