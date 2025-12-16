#!/usr/bin/env python3
"""
Fetch data from Bureau of Transportation Statistics and FRED.

This script downloads the necessary data for the gas prices vs unemployment
driving analysis, including:
- Highway Vehicle Miles Traveled
- Highway Fuel Price (Regular Gasoline)
- Unemployment Rate (from FRED)
- Real GDP (from FRED)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

try:
    from pandas_datareader import data as web
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("pandas_datareader not available. FRED data fetching disabled.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_numeric(series: pd.Series) -> pd.Series:
    """
    Remove currency symbols, commas, percent signs and convert to numeric.
    
    Args:
        series: Series with potentially formatted numeric values
        
    Returns:
        Series with numeric values
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False),
        errors='coerce'
    )

def fetch_fred_data(
    series_id: str,
    start_date: datetime,
    end_date: datetime,
    name: str
) -> pd.Series:
    """
    Fetch data from FRED (Federal Reserve Economic Data).
    
    Args:
        series_id: FRED series identifier
        start_date: Start date for data
        end_date: End date for data
        name: Name for the series
        
    Returns:
        Series with date index and values
    """
    if not FRED_AVAILABLE:
        raise ImportError("pandas_datareader is required for FRED data. Install with: pip install pandas-datareader")
    
    logging.info(f"Fetching {name} (FRED: {series_id})...")
    df = web.DataReader(series_id, 'fred', start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data returned for {series_id}")
    
    series = df[series_id].copy()
    series.name = name
    return series

def fetch_bts_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load BTS Monthly Transportation Statistics data.
    
    This function expects a CSV file downloaded from:
    https://www.bts.gov/content/monthly-transportation-statistics
    
    Args:
        filepath: Path to BTS CSV file. If None, looks in data/ and content/ directories.
        
    Returns:
        DataFrame with cleaned BTS data
    """
    if filepath is None:
        # Try multiple possible locations
        possible_paths = [
            Path("data/Monthly_Transportation_Statistics.csv"),
            Path("content/Monthly_Transportation_Statistics_20250419.csv"),
            Path("data/Monthly_Transportation_Statistics_20250419.csv"),
        ]
        
        for path in possible_paths:
            if path.exists():
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError(
                "BTS data file not found. Please download from "
                "https://www.bts.gov/content/monthly-transportation-statistics"
            )
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"BTS data file not found at {filepath}. "
            "Please download from https://www.bts.gov/content/monthly-transportation-statistics"
        )
    
    logging.info(f"Loading BTS data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Expected column names (adjust if BTS changes format)
    required_cols = [
        'Date',
        'Highway Vehicle Miles Traveled - All Systems',
        'Highway Fuel Price - Regular Gasoline',
    ]
    
    # Check if columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.warning(f"Missing columns: {missing_cols}")
        logging.info(f"Available columns: {list(df.columns)[:10]}...")
        raise ValueError(f"Required columns not found: {missing_cols}")
    
    # Select and rename columns
    df = df[required_cols].copy()
    df.columns = ['Date', 'Miles_Traveled', 'Gas_Price']
    
    # Parse dates - handle multiple date formats
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    if df['Date'].isna().any():
        # Try alternative format
        df['Date'] = pd.to_datetime(df['Date'], format='%Y %b %d %I:%M:%S %p', errors='coerce')
    
    # Clean numeric columns
    df['Miles_Traveled'] = clean_numeric(df['Miles_Traveled'])
    df['Gas_Price'] = clean_numeric(df['Gas_Price'])
    
    # Sort and remove invalid dates
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=['Date'])
    
    # Filter to analysis period (2018-2025)
    start_date = pd.to_datetime('2018-01-01')
    end_date = pd.to_datetime('2025-12-31')
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    logging.info(f"Filtered to {len(df)} observations from {df['Date'].min()} to {df['Date'].max()}")
    
    return df

def combine_data(
    bts_df: pd.DataFrame,
    unemployment: pd.Series,
    gdp: pd.Series
) -> pd.DataFrame:
    """
    Combine BTS data with FRED data into a single DataFrame.
    
    Args:
        bts_df: DataFrame with BTS data (Date, Miles_Traveled, Gas_Price)
        unemployment: Series with unemployment rate (date index)
        gdp: Series with GDP (date index)
        
    Returns:
        Combined DataFrame with all variables
    """
    # Start with BTS data
    df = bts_df.copy()
    
    # Merge unemployment
    unemployment_df = unemployment.reset_index()
    unemployment_df.columns = ['Date', 'Unemployment']
    df = df.merge(unemployment_df, on='Date', how='left')
    
    # Merge GDP (quarterly, so forward fill)
    gdp_df = gdp.reset_index()
    gdp_df.columns = ['Date', 'GDP']
    
    # Forward fill GDP to monthly frequency
    df = df.merge(gdp_df, on='Date', how='left')
    df['GDP'] = df['GDP'].ffill()
    
    # Remove rows with missing data
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    if initial_len != final_len:
        logging.info(f"Dropped {initial_len - final_len} rows with missing data")
    
    return df

def fetch_all_data(
    start_date: datetime = datetime(2018, 1, 1),
    end_date: datetime = datetime(2025, 8, 31),
    bts_filepath: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch all data needed for the analysis.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        bts_filepath: Path to BTS CSV file
        output_path: Optional path to save combined data
        
    Returns:
        DataFrame with all variables
    """
    logging.info("=" * 70)
    logging.info("Fetching data for Gas Prices vs Unemployment Driving Analysis")
    logging.info("=" * 70)
    
    # Fetch BTS data
    bts_df = fetch_bts_data(bts_filepath)
    logging.info(f"BTS data: {len(bts_df)} observations")
    logging.info(f"Date range: {bts_df['Date'].min()} to {bts_df['Date'].max()}")
    
    # Fetch FRED data
    if FRED_AVAILABLE:
        unemployment = fetch_fred_data('UNRATE', start_date, end_date, 'Unemployment')
        gdp = fetch_fred_data('GDPC1', start_date, end_date, 'Real GDP')
        
        logging.info(f"Unemployment data: {len(unemployment)} observations")
        logging.info(f"GDP data: {len(gdp)} observations")
    else:
        logging.warning("FRED data not available. Using BTS data only.")
        # Create dummy series for unemployment and GDP
        unemployment = pd.Series(index=bts_df['Date'], name='Unemployment')
        gdp = pd.Series(index=bts_df['Date'], name='GDP')
    
    # Combine data
    df = combine_data(bts_df, unemployment, gdp)
    
    logging.info(f"Combined dataset: {len(df)} observations")
    logging.info(f"Variables: {list(df.columns)}")
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Data saved to {output_path}")
    
    return df

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch data for gas prices analysis')
    parser.add_argument('--bts-file', type=Path, default=None,
                       help='Path to BTS CSV file')
    parser.add_argument('--output', type=Path, default='data/combined_data.csv',
                       help='Output path for combined data')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-08-31',
                       help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    df = fetch_all_data(
        start_date=start_date,
        end_date=end_date,
        bts_filepath=args.bts_file,
        output_path=args.output
    )
    
    logging.info("\nData Summary:")
    logging.info(df.describe())
    logging.info(f"\nData saved to: {args.output}")

if __name__ == '__main__':
    main()

