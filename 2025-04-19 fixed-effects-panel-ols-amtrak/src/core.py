"""Core functions for fixed effects panel OLS modeling."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def prepare_panel_data(df: pd.DataFrame, entity_col: str, time_col: str, 
                       value_col: str) -> pd.DataFrame:
    """Prepare data for panel analysis."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col]) if not pd.api.types.is_datetime64_any_dtype(df[time_col]) else df[time_col]
    return df


def fit_fixed_effects_model(df: pd.DataFrame, formula: str, entity_col: str) -> any:
    """Fit fixed effects panel OLS model."""
    df_fe = df.copy()
    df_fe = df_fe.set_index([entity_col])
    model = smf.ols(formula, data=df_fe).fit(cov_type='cluster', cov_kwds={'groups': df_fe.index})
    return model


def fit_panel_ols(df: pd.DataFrame, outcome: str, features: List[str],
                  entity_col: str, time_col: str) -> any:
    """Fit panel OLS with entity and time fixed effects."""
    formula = f"{outcome} ~ {' + '.join(features)}"
    if entity_col:
        formula += f" + C({entity_col})"
    if time_col:
        formula += f" + C({time_col})"
    
    model = smf.ols(formula, data=df).fit()
    return model


def plot_panel_results(df: pd.DataFrame, entity_col: str, value_col: str,
                      title: str, output_path: Path):
    """Plot panel data results with Tufte style."""
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_df = df.pivot_table(index=df.columns[1] if len(df.columns) > 1 else df.index,
                              columns=entity_col, values=value_col)
    
    for col in pivot_df.columns[:10]:
        ax.plot(pivot_df.index, pivot_df[col], alpha=0.6, linewidth=1.2)
    
    apply_tufte_style(ax, title=title)
    ax.set_xlabel("Time")
    ax.set_ylabel(value_col)
    
    save_tufte_figure(output_path)

