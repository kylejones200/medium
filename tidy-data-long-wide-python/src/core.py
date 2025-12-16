"""Core functions for tidy data transformations (long/wide format)."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_wide_data(stores: list = None, months: list = None) -> pd.DataFrame:
    """Generate wide format data."""
    if stores is None:
        stores = ['A', 'B']
    if months is None:
        months = ['Jan_Sales', 'Feb_Sales', 'Mar_Sales']
    
    data = {'Store': stores}
    for i, month in enumerate(months):
        data[month] = [100 + i*10, 90 + i*10]
    return pd.DataFrame(data)

def wide_to_long(df: pd.DataFrame, id_vars: str, var_name: str = 'Month', 
                value_name: str = 'Sales', suffix: str = '_Sales') -> pd.DataFrame:
    """Convert wide format to long format."""
    long_df = pd.melt(df, id_vars=id_vars, var_name=var_name, value_name=value_name)
    if suffix:
        long_df[var_name] = long_df[var_name].str.replace(suffix, '')
    return long_df

def long_to_wide(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    """Convert long format to wide format."""
    return df.pivot(index=index, columns=columns, values=values).reset_index()

def pivot_table_aggregation(df: pd.DataFrame, index: str, columns: str, 
                           values: str, aggfunc: str = 'sum') -> pd.DataFrame:
    """Create pivot table with aggregation."""
    return df.pivot_table(index=index, columns=columns, values=values, 
                         aggfunc=aggfunc).reset_index()

def groupby_aggregation(df: pd.DataFrame, groupby_cols: list, value_col: str,
                        aggfuncs: Dict[str, str]) -> pd.DataFrame:
    """Perform groupby aggregation with multiple functions."""
    return df.groupby(groupby_cols)[value_col].agg(aggfuncs).reset_index()

def generate_weekly_sales_data(stores: list = None, weeks: int = 3) -> pd.DataFrame:
    """Generate weekly sales data in wide format."""
    if stores is None:
        stores = ['North', 'South', 'East', 'West']
    
    data = {'Store': stores}
    for week in range(1, weeks + 1):
        data[f'Week_{week}'] = [300 + week*10, 250 + week*5, 400 - week*5, 375 + week*3]
    return pd.DataFrame(data)

def reshape_weekly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape weekly data from wide to long format."""
    long = pd.melt(df, id_vars='Store', var_name='Week', value_name='Sales')
    long['Week'] = long['Week'].str.replace('Week_', '').astype(int)
    return long

def plot_weekly_trend(df: pd.DataFrame, output_path: Path):
 """Plot weekly sales trend """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weekly = df.groupby('Week')['Sales'].sum().reset_index()
    ax.plot(weekly['Week'], weekly['Sales'], marker='o', color="#4A90A4", linewidth=1.2)
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Sales')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

def plot_store_comparison(df: pd.DataFrame, output_path: Path):
 """Plot store-wise comparison """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot(index='Week', columns='Store', values='Sales').reset_index()
    colors = ["#4A90A4", "#D4A574", "#8B6F9E", "#A8C5A0"]
    for i, store in enumerate(pivot.columns[1:]):
        ax.plot(pivot['Week'], pivot[store], label=store, 
               color=colors[i % len(colors)], linewidth=1.2, marker='o')
    
    ax.set_xlabel('Week')
    ax.set_ylabel('Sales')
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
