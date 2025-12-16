"""Core functions for integrating Excel and Python for business analytics."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def read_excel_data(file_path: Path, sheet_name: str = None) -> pd.DataFrame:
    """Read data from Excel file."""
    if sheet_name:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    return pd.read_excel(file_path)

def write_excel_data(df: pd.DataFrame, file_path: Path, sheet_name: str = 'Sheet1'):
    """Write data to Excel file."""
    df.to_excel(file_path, sheet_name=sheet_name, index=False)

def analyze_excel_data(df: pd.DataFrame) -> Dict:
    """Analyze Excel data."""
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }

def plot_excel_analysis(df: pd.DataFrame, numeric_col: str, title: str, output_path: Path):
 """Plot Excel data analysis """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if numeric_col in df.columns and df[numeric_col].dtype in ['int64', 'float64']:
        ax.plot(df.index, df[numeric_col], color="#4A90A4", linewidth=1.2)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            ax.plot(df.index, df[numeric_cols[0]], color="#4A90A4", linewidth=1.2)
        else:
            ax.text(0.5, 0.5, 'No numeric data to plot', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

