"""Core functions for dummy variables in econometric policy analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_dummy_variables(df: pd.DataFrame, date_col: str, policy_date: str) -> pd.DataFrame:
    """Create dummy variable for policy intervention."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    policy_date = pd.to_datetime(policy_date)
    df['policy_dummy'] = (df[date_col] >= policy_date).astype(int)
    return df

def fit_policy_regression(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> sm.OLS:
    """Fit OLS regression with dummy variables."""
    X = df[x_cols]
    X = sm.add_constant(X)
    y = df[y_col]
    model = sm.OLS(y, X)
    return model.fit()

def plot_policy_effect(df: pd.DataFrame, y_col: str, date_col: str, policy_date: str,
                      title: str, output_path: Path):
 """Plot policy effect """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df[date_col] = pd.to_datetime(df[date_col])
    policy_date = pd.to_datetime(policy_date)
    
    pre_policy = df[df[date_col] < policy_date]
    post_policy = df[df[date_col] >= policy_date]
    
    ax.plot(pre_policy[date_col], pre_policy[y_col], label="Pre-Policy", 
           color="#4A90A4", linewidth=1.2)
    ax.plot(post_policy[date_col], post_policy[y_col], label="Post-Policy", 
           color="#D4A574", linewidth=1.2)
    ax.axvline(policy_date, color='red', linestyle='--', linewidth=1.2, label='Policy Date')
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

