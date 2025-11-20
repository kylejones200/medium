"""Core functions for Regression Discontinuity (RD) analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.formula.api as smf
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure


def fetch_fred_data(series: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch data from FRED for multiple series."""
    dfs = []
    for name, code in series.items():
        df = web.DataReader(code, "fred", start=start_date, end=end_date)
        dfs.append(df)
    
    data = dfs[0]
    for df in dfs[1:]:
        data = data.join(df)
    
    return data


def compute_inflation(cpi_data: pd.DataFrame, periods: int = 12) -> pd.DataFrame:
    """Compute year-over-year inflation."""
    cpi_data = cpi_data.copy()
    cpi_data['Inflation'] = cpi_data.iloc[:, 0].pct_change(periods) * 100
    return cpi_data[['Inflation']]


def create_rd_variables(data: pd.DataFrame, cutoff_date: str, 
                      date_col: str = 'DATE', bandwidth: int = 30) -> pd.DataFrame:
    """Create running variable and treatment indicators for RD design."""
    data = data.copy()
    cutoff = pd.to_datetime(cutoff_date)
    
    data['Months'] = (data[date_col].dt.to_period('M').astype(int) - 
                     cutoff.to_period('M').ordinal)
    data = data[(data['Months'] >= -bandwidth) & (data['Months'] <= bandwidth)].copy()
    data['Treatment'] = (data['Months'] >= 0).astype(int)
    data['Centered_Months'] = data['Months']
    
    return data


def fit_rd_model(data: pd.DataFrame, outcome: str, bandwidth: int,
                formula: str = None) -> any:
    """Fit regression discontinuity model."""
    if formula is None:
        formula = f'{outcome} ~ Centered_Months * Treatment'
    
    subset = data[(data['Months'] >= -bandwidth) & (data['Months'] <= bandwidth)]
    model = smf.ols(formula, data=subset).fit()
    return model


def fit_placebo_model(data: pd.DataFrame, outcome: str, placebo_cutoff: str) -> any:
    """Fit placebo RD model with alternative cutoff."""
    data = data.copy()
    placebo_date = pd.to_datetime(placebo_cutoff)
    data['Placebo_Months'] = (data['DATE'].dt.to_period('M').astype(int) - 
                              placebo_date.to_period('M').ordinal)
    data['Placebo_Treatment'] = (data['Placebo_Months'] >= 0).astype(int)
    data['Centered_Placebo'] = data['Placebo_Months']
    
    formula = f'{outcome} ~ Centered_Placebo * Placebo_Treatment'
    model = smf.ols(formula, data=data).fit()
    return model


def plot_rd_design(x: pd.Series, y: pd.Series, title: str, ylabel: str,
                  output_path: Path, ylim_zero: bool = False):
    """Plot regression discontinuity design with Tufte style."""
    setup_tufte_style()
    
    lowess_fit = lowess(y, x, frac=0.3)
    lowess_x = lowess_fit[:, 0]
    lowess_y = lowess_fit[:, 1]
    interp_y = np.interp(x, lowess_x, lowess_y)
    residuals = y - interp_y
    rolling_std = pd.Series(residuals).rolling(window=3, center=False).shift(1).std()
    interp_std = np.interp(lowess_x, x, rolling_std)
    upper = lowess_y + 1.96 * interp_std
    lower = lowess_y - 1.96 * interp_std
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='#4A90A4', s=12, alpha=0.5)
    ax.plot(lowess_x, lowess_y, color='#4A90A4', linewidth=1.5)
    ax.fill_between(lowess_x, lower, upper, color='#4A90A4', alpha=0.1)
    ax.axvline(0, color='#D4A574', linestyle='--', linewidth=1.2)
    
    apply_tufte_style(ax, title=title)
    ax.set_xlabel('Months from Cutoff', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if ylim_zero:
        ax.set_ylim(bottom=0)
    
    save_tufte_figure(output_path)


def create_summary_table(models: Dict[str, any], treatment_params: Dict[str, str]) -> pd.DataFrame:
    """Create summary table of treatment effects."""
    results = []
    for name, model in models.items():
        param_name = treatment_params.get(name, 'Treatment')
        if param_name in model.params:
            results.append({
                'Model': name,
                'Treatment Effect': model.params[param_name],
                'p-value': model.pvalues[param_name]
            })
    
    return pd.DataFrame(results)

