#!/usr/bin/env python3
"""
Time Series Analysis: Unemployment Rate Forecasting and Natural Gas Bollinger Bands

This script demonstrates:
1. Loading economic data from FRED (Federal Reserve Economic Data)
2. Forecasting unemployment rates using Prophet
3. Calculating and visualizing Bollinger Bands for natural gas prices

Author: K.T. Jones
Date: 2024-10-23
"""

import datetime
import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator
from pandas_datareader import data as web
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_fred_data(
    series_id: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    rename_columns: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load economic data from FRED (Federal Reserve Economic Data).
    
    Args:
        series_id: FRED series identifier (e.g., 'UNRATE', 'DHHNGSP')
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        rename_columns: Optional dictionary to rename columns
        
    Returns:
        DataFrame with datetime index and data columns
        
    Examples:
        >>> start = datetime.datetime(2010, 1, 1)
        >>> end = datetime.datetime(2025, 2, 20)
        >>> df = load_fred_data('UNRATE', start, end)
    """
    df = web.DataReader(series_id, 'fred', start_date, end_date)
    df.reset_index(inplace=True)
    
    if rename_columns:
        df.rename(columns=rename_columns, inplace=True)
    else:
        # Default: rename DATE to ds and series_id to y for Prophet compatibility
        if 'DATE' in df.columns:
            df.rename(columns={'DATE': 'ds'}, inplace=True)
        if series_id in df.columns:
            df.rename(columns={series_id: 'y'}, inplace=True)
    
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna().sort_values(by='ds').reset_index(drop=True)
    
    return df

# ============================================================================
# Prophet Forecasting Functions
# ============================================================================

def forecast_with_prophet(
    df: pd.DataFrame,
    periods: int = 12,
    freq: str = 'MS',
    date_col: str = 'ds',
    value_col: str = 'y'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate forecasts using Facebook Prophet.
    
    Args:
        df: DataFrame with datetime and value columns
        periods: Number of periods to forecast
        freq: Frequency string (e.g., 'MS' for month start)
        date_col: Name of date column
        value_col: Name of value column
        
    Returns:
        Tuple of (forecast DataFrame, Prophet model)
    """
    # Prepare data for Prophet
    prophet_df = df[[date_col, value_col]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Fit model
    model = Prophet()
    model.fit(prophet_df)
    
    # Generate future dates and forecast
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    return forecast, model

def plot_prophet_forecast(
    model: Prophet,
    forecast: pd.DataFrame,
    title: str = "Prophet Forecast",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot Prophet forecast with uncertainty bands.
    
    Args:
        model: Fitted Prophet model
        forecast: Forecast DataFrame from model.predict()
        title: Plot title
        output_path: Optional path to save figure
    """
    fig = model.plot(forecast)
    ax = plt.gca()
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# ============================================================================
# Bollinger Bands Functions
# ============================================================================

def calculate_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    target_col: str = 'adjClose',
    dropna: bool = True
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a time series.
    
    Bollinger Bands consist of:
    - Middle band: N-period moving average
    - Upper band: Middle band + (N-period std dev * num_std)
    - Lower band: Middle band - (N-period std dev * num_std)
    
    Args:
        df: DataFrame with time series data
        window: Rolling window size (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
        target_col: Column name to calculate bands for
        dropna: Whether to drop rows with NaN values
        
    Returns:
        DataFrame with added Bollinger Band columns
        
    Examples:
        >>> df = calculate_bollinger_bands(df, window=20, num_std=2.0)
    """
    df = df.copy()
    
    # Calculate moving average
    ma_col = f'{window} Day MA'
    df[ma_col] = df[target_col].rolling(window=window).mean()
    
    # Calculate standard deviation
    std = df[target_col].rolling(window=window).std()
    
    # Calculate bands
    df[f'{ma_col}_lower'] = df[ma_col] - (num_std * std)
    df[f'{ma_col}_upper'] = df[ma_col] + (num_std * std)
    
    if dropna:
        df = df.dropna()
    
    return df

def plot_bollinger_bands(
    df: pd.DataFrame,
    target_col: str = 'adjClose',
    window: int = 20,
    title: Optional[str] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot time series with Bollinger Bands.
    
    Args:
        df: DataFrame with Bollinger Bands calculated
        target_col: Column name to plot
        window: Window size used for calculation
        title: Optional plot title
        output_path: Optional path to save figure
    """
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'ds' in df.columns:
            df = df.set_index('ds')
        elif 'DATE' in df.columns:
            df = df.set_index('DATE')
        else:
            raise ValueError("DataFrame must have datetime index or 'ds'/'DATE' column")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ma_col = f'{window} Day MA'
    lower_col = f'{ma_col}_lower'
    upper_col = f'{ma_col}_upper'
    
    # Fill between bands
    ax.fill_between(
        df.index,
        df[lower_col].values,
        df[upper_col].values,
        alpha=0.15,
        color="#8B6F9E",
        label="Bollinger Band"
    )
    
    # Plot actual values
    ax.plot(
        df.index,
        df[target_col].values,
        label=target_col,
        color="#4A90A4",
        linewidth=1.2
    )
    
    # Plot moving average
    ax.plot(
        df.index,
        df[ma_col].values,
        label=f"{window} Day MA",
        color="#D4A574",
        linewidth=1.2,
        linestyle="--"
    )
    
    # Apply styling
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc='best')
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation=45)
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# ============================================================================
# Utility Plotting Functions
# ============================================================================

def plot_time_series_simple(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: Optional[str] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot a simple time series.
    
    Args:
        df: DataFrame with time series data
        date_col: Name of date column
        value_col: Name of value column
        title: Optional plot title
        output_path: Optional path to save figure
    """
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col, value_col])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[date_col], df[value_col], color='black', linewidth=2)
    
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    
    # Format x-axis
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation=45)
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# ============================================================================
# Main Analysis Functions
# ============================================================================

def analyze_unemployment_rate(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    forecast_periods: int = 12,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Prophet]:
    """
    Analyze unemployment rate data with Prophet forecasting.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        forecast_periods: Number of periods to forecast
        output_dir: Optional directory to save plots
        
    Returns:
        Tuple of (data DataFrame, forecast DataFrame, Prophet model)
    """
    logging.info("Loading unemployment rate data from FRED...")
    df = load_fred_data('UNRATE', start_date, end_date)
    logging.info(f"Loaded {len(df)} observations")
    
    logging.info("Fitting Prophet model...")
    forecast, model = forecast_with_prophet(df, periods=forecast_periods)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plot_prophet_forecast(
            model,
            forecast,
            title="Unemployment Rate Forecast using Prophet",
            output_path=output_dir / "unemployment_prophet_forecast.png"
        )
    else:
        plot_prophet_forecast(
            model,
            forecast,
            title="Unemployment Rate Forecast using Prophet"
        )
    
    return df, forecast, model

def analyze_natural_gas_bollinger_bands(
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    window: int = 20,
    num_std: float = 2.0,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Analyze natural gas prices with Bollinger Bands.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        window: Rolling window for Bollinger Bands
        num_std: Number of standard deviations for bands
        output_dir: Optional directory to save plots
        
    Returns:
        DataFrame with Bollinger Bands calculated
    """
    logging.info("Loading natural gas spot price data from FRED...")
    df = load_fred_data('DHHNGSP', start_date, end_date)
    
    # Set up for Bollinger Bands (use ds as index, rename y to adjClose)
    df = df.set_index('ds')
    df['adjClose'] = df['y']
    df = df[['adjClose']]
    
    logging.info(f"Loaded {len(df)} observations")
    logging.info("Calculating Bollinger Bands...")
    df = calculate_bollinger_bands(df, window=window, num_std=num_std)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plot_bollinger_bands(
            df,
            target_col='adjClose',
            window=window,
            title="Natural Gas Spot Price - Bollinger Bands",
            output_path=output_dir / "natural_gas_bollinger_bands.png"
        )
    else:
        plot_bollinger_bands(
            df,
            target_col='adjClose',
            window=window,
            title="Natural Gas Spot Price - Bollinger Bands"
        )
    
    return df

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    # Create output directory
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logging.info("Time Series Analysis: Unemployment & Natural Gas Bollinger Bands")
    
    # Analysis 1: Unemployment Rate Forecasting
    logging.info("Analysis 1: Unemployment Rate Forecasting")
    unemployment_start = datetime.datetime(2010, 1, 1)
    unemployment_end = datetime.datetime(2025, 2, 20)
    
    try:
        df_unemployment, forecast_unemployment, model_unemployment = analyze_unemployment_rate(
            unemployment_start,
            unemployment_end,
            forecast_periods=12,
            output_dir=output_dir
        )
        logging.info("✓ Unemployment rate analysis complete")
        logging.info(f"  Forecast range: {forecast_unemployment['ds'].min()} to {forecast_unemployment['ds'].max()}")
    except Exception as e:
        logging.error(f"✗ Error in unemployment analysis: {e}")
        df_unemployment = None
    
    # Analysis 2: Natural Gas Bollinger Bands
    logging.info("Analysis 2: Natural Gas Bollinger Bands")
    gas_start = datetime.datetime(2024, 4, 1)
    gas_end = datetime.datetime(2024, 10, 20)
    
    try:
        df_gas = analyze_natural_gas_bollinger_bands(
            gas_start,
            gas_end,
            window=20,
            num_std=2.0,
            output_dir=output_dir
        )
        logging.info("✓ Natural gas Bollinger Bands analysis complete")
        logging.info(f"  Data range: {df_gas.index.min()} to {df_gas.index.max()}")
        logging.info(f"  Current price: ${df_gas['adjClose'].iloc[-1]:.2f}")
        logging.info(f"  20-day MA: ${df_gas['20 Day MA'].iloc[-1]:.2f}")
    except Exception as e:
        logging.error(f"✗ Error in natural gas analysis: {e}")
        df_gas = None
    
    logging.info(f"Analysis complete! Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

