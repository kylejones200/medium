"""Tests for core time series analysis functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.core import (
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

    generate_synthetic_data,
    split_data,
    test_stationarity,
    calculate_mape
)

def test_generate_synthetic_data():
    """Test synthetic data generation."""
    df = generate_synthetic_data(n_samples=100, seed=42)
    assert len(df) == 100
    assert 'value' in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)

def test_split_data():
    """Test data splitting."""
    df = generate_synthetic_data(n_samples=100, seed=42)
    train, hold_out = split_data(df, hold_out_days=20)
    assert len(train) == 80
    assert len(hold_out) == 20
    assert len(train) + len(hold_out) == len(df)

def test_stationarity():
    """Test stationarity test."""
    df = generate_synthetic_data(n_samples=100, seed=42)
    result = test_stationarity(df["value"])
    assert 'adf_statistic' in result
    assert 'p_value' in result
    assert 'is_stationary' in result
    assert isinstance(result['is_stationary'], bool)

def test_calculate_mape():
    """Test MAPE calculation."""
    actual = pd.Series([10, 20, 30, 40, 50])
    predicted = pd.Series([11, 19, 31, 39, 51])
    mape = calculate_mape(actual, predicted)
    assert mape >= 0
    assert isinstance(mape, float)

if __name__ == "__main__":
    pytest.main([__file__])

