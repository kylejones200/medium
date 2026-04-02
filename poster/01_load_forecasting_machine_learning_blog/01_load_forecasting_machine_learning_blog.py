#!/usr/bin/env python3
"""
Python code extracted from 01_load_forecasting_machine_learning_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EIAParquetService:
    """Service for handling the massive EIA electricity parquet dataset."""
    
    def __init__(self, parquet_path: str = "ELEC.parquet"):
        """Initialize EIA parquet service.
        
        Args:
            parquet_path: Path to the ELEC.parquet file.
        """
        self.parquet_path = parquet_path
        self.raw_data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and parse the EIA parquet data."""
        try:
            logger.info(f"Loading EIA parquet data from {self.parquet_path}")
            self.raw_data = pd.read_parquet(self.parquet_path)
            logger.info(f"Loaded {len(self.raw_data)} EIA records")
            
        except Exception as e:
            logger.error(f"Failed to load EIA parquet data: {e}")
            self.raw_data = pd.DataFrame()
    
    def search_series(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for series matching a query."""
        if self.raw_data is None or self.raw_data.empty:
            return []
        
        results = []
        column_name = self.raw_data.columns[0]
        
        # Search through records
        for i, row in self.raw_data.iterrows():
            if len(results) >= limit:
                break
                
            json_str = row[column_name]
            if query.lower() in json_str.lower():
                try:
                    parsed = json.loads(json_str)
                    results.append({
                        "series_id": parsed.get('series_id', ''),
                        "name": parsed.get('name', ''),
                        "units": parsed.get('units', ''),
                        "geography": parsed.get('geography', ''),
                        "start": parsed.get('start', ''),
                        "end": parsed.get('end', ''),
                        "data_points": len(parsed.get('data', []))
                    })
                except json.JSONDecodeError:
                    continue
        
        return results
    
    def get_time_series_data(self, series_id: str) -> Dict[str, Any]:
        """Get complete time series data for a specific series."""
        if self.raw_data is None or self.raw_data.empty:
            return {}
        
        column_name = self.raw_data.columns[0]
        
        for i, row in self.raw_data.iterrows():
            json_str = row[column_name]
            try:
                parsed = json.loads(json_str)
                if parsed.get('series_id') == series_id:
                    # Convert data to proper time series format
                    data_points = parsed.get('data', [])
                    
                    dates = []
                    values = []
                    
                    for point in data_points:
                        if len(point) >= 2:
                            date_str = point[0]
                            value = point[1]
                            
                            # Convert YYYYMM to proper date
                            if len(date_str) == 6:
                                year = int(date_str[:4])
                                month = int(date_str[4:])
                                dates.append(f"{year}-{month:02d}-01")
                                values.append(value)
                    
                    return {
                        "series_id": series_id,
                        "name": parsed.get('name', ''),
                        "units": parsed.get('units', ''),
                        "dates": dates,
                        "values": values,
                        "metadata": {
                            "lat": parsed.get('lat'),
                            "lon": parsed.get('lon'),
                            "geography": parsed.get('geography'),
                            "start": parsed.get('start'),
                            "end": parsed.get('end')
                        }
                    }
                    
            except json.JSONDecodeError:
                continue
        
        return {}

# Example usage: Load California generation data
eia_service = EIAParquetService("ELEC.parquet")
results = eia_service.search_series("generation CAL", limit=10)

for result in results:
    print(f"Series: {result['series_id']}")
    print(f"Name: {result['name']}")
    print(f"Data points: {result['data_points']}")
    print()

# Get detailed time series for best matching series
if results:
    series_data = eia_service.get_time_series_data(results[0]['series_id'])
    print(f"Retrieved {len(series_data['values'])} monthly data points")
    print(f"Date range: {series_data['dates'][0]} to {series_data['dates'][-1]}")

# ======================================================================
# Code Block 2
# ======================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

def prepare_features(load_data: pd.DataFrame, lookback_days: int = 90) -> pd.DataFrame:
    """Prepare feature dataset for modeling.
    
    Args:
        load_data: DataFrame with columns [ts_utc, mw, ba]
        lookback_days: Number of days of historical data to use.
        
    Returns:
        DataFrame with engineered features.
    """
    df = load_data.copy()
    df['ts_utc'] = pd.to_datetime(df['ts_utc'])
    df = df.sort_values('ts_utc').reset_index(drop=True)
    
    # Calendar features
    df['hour'] = df['ts_utc'].dt.hour
    df['dow'] = df['ts_utc'].dt.dayofweek + 1  # 1=Monday
    df['month'] = df['ts_utc'].dt.month
    df['day_of_year'] = df['ts_utc'].dt.dayofyear
    df['is_weekend'] = (df['dow'] >= 6).astype(int)
    
    # Cyclical encoding for hour (captures 23->0 continuity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of year (captures Dec 31 -> Jan 1)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Lag features (key for time series forecasting)
    df['mw_lag1'] = df['mw'].shift(1)      # 1 hour ago
    df['mw_lag24'] = df['mw'].shift(24)    # Same hour yesterday
    df['mw_lag168'] = df['mw'].shift(168)  # Same hour last week
    
    # Rolling statistics (capture recent trends)
    df['mw_ma24'] = df['mw'].rolling(window=24, min_periods=1).mean()
    df['mw_ma168'] = df['mw'].rolling(window=168, min_periods=1).mean()
    df['mw_std24'] = df['mw'].rolling(window=24, min_periods=1).std()
    
    # Temperature features (synthetic if real weather unavailable)
    df['temperature'] = 70 + 20 * np.sin(2 * np.pi * df['day_of_year'] / 365.25) + \
                       10 * np.sin(2 * np.pi * df['hour'] / 24)
    df['temp_squared'] = df['temperature'] ** 2
    df['cooling_degree_days'] = np.maximum(df['temperature'] - 65, 0)
    df['heating_degree_days'] = np.maximum(55 - df['temperature'], 0)
    
    # Holiday indicator (simplified - enhance with actual holiday calendar)
    df['is_holiday'] = 0
    
    return df

# Example: Prepare features for California
sample_data = pd.DataFrame({
    'ts_utc': pd.date_range('2024-01-01', periods=24*90, freq='H'),
    'mw': np.random.normal(25000, 5000, 24*90),  # Synthetic load
    'ba': 'CAL-ALL'
})

features_df = prepare_features(sample_data)
print(f"Original columns: {len(sample_data.columns)}")
print(f"Feature columns: {len(features_df.columns)}")
print(f"\nNew features: {list(features_df.columns[3:])}")

# ======================================================================
# Code Block 3
# ======================================================================

from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import mlflow
import mlflow.sklearn

def train_arima_model(df: pd.DataFrame, ba: str) -> Optional[str]:
    """Train auto_arima baseline model.
    
    Args:
        df: Feature DataFrame.
        ba: Balancing authority code.
        
    Returns:
        Model URI if successful, None otherwise.
    """
    # Prepare time series data
    ts_data = df[['ts_utc', 'mw']].copy()
    ts_data = ts_data.dropna().sort_values('ts_utc')
    
    if len(ts_data) < 168:  # Need at least 1 week of data
        print(f"Insufficient data: {len(ts_data)} records")
        return None
    
    # Set datetime index
    ts_data.set_index('ts_utc', inplace=True)
    ts_series = ts_data['mw']
    
    with mlflow.start_run(run_name=f"arima_{ba}"):
        # Use auto_arima to find optimal parameters
        print(f"Training auto_arima model for {ba}...")
        
        model = auto_arima(
            ts_series,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            seasonal=True,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            m=24,  # 24-hour seasonality
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        # Make in-sample predictions
        fitted_values = model.fittedvalues()
        actual_values = ts_series[fitted_values.index]
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(actual_values, fitted_values)
        mae = mean_absolute_error(actual_values, fitted_values)
        
        # Log to MLflow
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("model_type", "auto_arima")
        mlflow.log_param("order", str(model.order))
        mlflow.log_param("seasonal_order", str(model.seasonal_order))
        
        # Save model
        model_info = mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"leap_{ba}_arima"
        )
        
        print(f"ARIMA model trained: MAPE={mape:.4f}, Order={model.order}")
        return model_info.model_uri

# Example usage
sample_features = prepare_features(sample_data)
model_uri = train_arima_model(sample_features, "CAL-ALL")

# ======================================================================
# Code Block 4
# ======================================================================

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

def train_lightgbm_model(df: pd.DataFrame, ba: str) -> Optional[str]:
    """Train LightGBM advanced model.
    
    Args:
        df: Feature DataFrame.
        ba: Balancing authority code.
        
    Returns:
        Model URI if successful, None otherwise.
    """
    # Select features
    feature_cols = [
        'mw_lag1', 'mw_lag24', 'mw_lag168',
        'mw_ma24', 'mw_ma168', 'mw_std24',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'dow', 'month', 'is_weekend', 'is_holiday',
        'temperature', 'temp_squared',
        'cooling_degree_days', 'heating_degree_days'
    ]
    
    # Filter to complete cases
    model_df = df[feature_cols + ['mw']].dropna()
    
    if len(model_df) < 168:
        print(f"Insufficient data: {len(model_df)} records")
        return None
    
    X = model_df[feature_cols]
    y = model_df['mw']
    
    with mlflow.start_run(run_name=f"lightgbm_{ba}"):
        # Time series cross-validation (respects temporal order)
        tscv = TimeSeriesSplit(n_splits=5)
        
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        # Cross-validation predictions
        cv_predictions = np.full(len(y), np.nan)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            cv_predictions[val_idx] = model.predict(X_val)
        
        # Calculate metrics
        valid_mask = ~np.isnan(cv_predictions)
        mape = mean_absolute_percentage_error(y[valid_mask], cv_predictions[valid_mask])
        mae = mean_absolute_error(y[valid_mask], cv_predictions[valid_mask])
        rmse = np.sqrt(mean_squared_error(y[valid_mask], cv_predictions[valid_mask]))
        
        # Train final model on all data
        model.fit(X, y)
        
        # Log to MLflow
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("model_type", "lightgbm")
        mlflow.log_param("n_estimators", 500)
        mlflow.log_param("learning_rate", 0.1)
        
        # Log feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_text(importance_df.to_string(), "feature_importance.txt")
        
        # Save model
        model_info = mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=f"leap_{ba}_lightgbm"
        )
        
        print(f"LightGBM trained: MAPE={mape:.4f}")
        print(f"\nTop 5 features:")
        print(importance_df.head())
        
        return model_info.model_uri

# Train LightGBM model
model_uri_lgbm = train_lightgbm_model(sample_features, "CAL-ALL")

# ======================================================================
# Code Block 5
# ======================================================================

def generate_forecast(model, df: pd.DataFrame, horizon_hours: int = 24) -> List[float]:
    """Generate multi-hour forecast using trained model.
    
    Args:
        model: Trained LightGBM model.
        df: Historical data with features.
        horizon_hours: Number of hours to forecast.
        
    Returns:
        List of forecast values.
    """
    feature_cols = [
        'mw_lag1', 'mw_lag24', 'mw_lag168',
        'mw_ma24', 'mw_ma168', 'mw_std24',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'dow', 'month', 'is_weekend', 'is_holiday',
        'temperature', 'temp_squared',
        'cooling_degree_days', 'heating_degree_days'
    ]
    
    # Get last complete row
    last_row = df.dropna(subset=feature_cols).iloc[-1].copy()
    forecasts = []
    
    for i in range(horizon_hours):
        # Update time-based features
        future_time = pd.to_datetime(last_row['ts_utc']) + timedelta(hours=i+1)
        hour = future_time.hour
        day_of_year = future_time.dayofyear
        
        last_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        last_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        last_row['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        last_row['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        last_row['dow'] = future_time.dayofweek + 1
        last_row['month'] = future_time.month
        last_row['is_weekend'] = int(last_row['dow'] >= 6)
        
        # Update temperature projection (simple seasonal model)
        last_row['temperature'] = 70 + 20 * np.sin(2 * np.pi * day_of_year / 365.25) + \
                                 10 * np.sin(2 * np.pi * hour / 24)
        last_row['temp_squared'] = last_row['temperature'] ** 2
        last_row['cooling_degree_days'] = max(last_row['temperature'] - 65, 0)
        last_row['heating_degree_days'] = max(55 - last_row['temperature'], 0)
        
        # Prepare features
        X = last_row[feature_cols].values.reshape(1, -1)
        
        # Make prediction
        forecast = model.predict(X)[0]
        forecasts.append(forecast)
        
        # Update lag features for next iteration
        last_row['mw_lag1'] = forecast
        last_row['mw_ma24'] = 0.95 * last_row['mw_ma24'] + 0.05 * forecast
    
    return forecasts

# Generate 48-hour forecast
forecasts_48h = generate_forecast(model, sample_features, horizon_hours=48)

print("48-Hour Load Forecast:")
for i, forecast in enumerate(forecasts_48h):
    print(f"Hour {i+1}: {forecast:,.0f} MW")

# ======================================================================
# Code Block 6
# ======================================================================

def apply_scenario(df: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    """Apply scenario adjustments to feature DataFrame.
    
    Args:
        df: Base features DataFrame.
        scenario_id: Scenario identifier.
        
    Returns:
        Modified DataFrame with scenario adjustments.
    """
    scenario_df = df.copy()
    
    if scenario_id == "hot_weather":
        # Model heat wave: +15°F temperature increase
        scenario_df['temperature'] += 15
        scenario_df['temp_squared'] = scenario_df['temperature'] ** 2
        scenario_df['cooling_degree_days'] = np.maximum(scenario_df['temperature'] - 65, 0)
        scenario_df['heating_degree_days'] = np.maximum(55 - scenario_df['temperature'], 0)
    
    elif scenario_id == "high_growth":
        # Model 5% load growth across all hours
        lag_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168']
        for col in lag_cols:
            if col in scenario_df.columns:
                scenario_df[col] *= 1.05
    
    elif scenario_id == "demand_response":
        # Model demand response program: reduce peak load 10%
        peak_hours = (scenario_df['hour'].between(16, 21))
        scenario_df.loc[peak_hours, 'mw_lag1'] *= 0.9
        scenario_df.loc[peak_hours, 'mw_lag24'] *= 0.9
    
    elif scenario_id == "major_outage":
        # Model major outage event: sudden 20% load drop
        scenario_df['mw_lag1'] *= 0.8
        scenario_df['mw_ma24'] *= 0.85  # Partial recovery
    
    return scenario_df

# Generate forecasts under different scenarios
scenarios = ["baseline", "hot_weather", "high_growth", "demand_response"]
scenario_forecasts = {}

for scenario in scenarios:
    if scenario == "baseline":
        scenario_features = sample_features.copy()
    else:
        scenario_features = apply_scenario(sample_features, scenario)
    
    forecasts = generate_forecast(model, scenario_features, horizon_hours=24)
    scenario_forecasts[scenario] = forecasts

# Compare scenarios
print("Peak Load Comparison (Hour 18):")
for scenario, forecasts in scenario_forecasts.items():
    peak_load = forecasts[17]  # Hour 18 (0-indexed)
    print(f"{scenario:20s}: {peak_load:>8,.0f} MW")

baseline_peak = scenario_forecasts["baseline"][17]
for scenario in scenarios[1:]:
    scenario_peak = scenario_forecasts[scenario][17]
    diff_pct = (scenario_peak / baseline_peak - 1) * 100
    print(f"  {scenario} vs baseline: {diff_pct:+.1f}%")

# ======================================================================
# Code Block 7
# ======================================================================

class OutageImpactAnalyzer:
    """Analyze how outages affect load forecasts."""
    
    def __init__(self, outage_service):
        self.outage_service = outage_service
    
    def get_outage_adjustment_factor(self, state: str, timestamp: datetime) -> float:
        """Calculate load adjustment factor based on current outages.
        
        Args:
            state: State code (e.g., 'Texas')
            timestamp: Current timestamp
            
        Returns:
            Adjustment factor (1.0 = no adjustment, 0.9 = 10% reduction)
        """
        # Get recent outage data
        year = timestamp.year
        outage_data = self.outage_service.get_outage_data(
            year=year,
            state=state,
            start_date=timestamp.strftime("%Y-%m-%d"),
            end_date=timestamp.strftime("%Y-%m-%d"),
            limit=100
        )
        
        if not outage_data:
            return 1.0
        
        # Calculate aggregate outage rate
        total_customers_out = sum(r['customers_out'] for r in outage_data)
        total_customers = sum(r.get('total_customers', 0) for r in outage_data if r.get('total_customers'))
        
        if total_customers == 0:
            return 1.0
        
        outage_rate = total_customers_out / total_customers
        
        # Load reduction roughly proportional to outage rate
        # But not perfectly linear (some load persists through outages)
        adjustment_factor = 1.0 - (outage_rate * 0.8)
        
        return max(0.5, adjustment_factor)  # Cap at 50% reduction
    
    def adjust_forecast_for_outages(self, forecast: List[float], state: str, 
                                    base_time: datetime) -> List[float]:
        """Adjust forecast based on expected or ongoing outages.
        
        Args:
            forecast: Base forecast values
            state: State code
            base_time: Base timestamp for forecast
            
        Returns:
            Adjusted forecast values
        """
        adjusted_forecast = []
        
        for i, forecast_value in enumerate(forecast):
            forecast_time = base_time + timedelta(hours=i)
            adjustment = self.get_outage_adjustment_factor(state, forecast_time)
            adjusted_forecast.append(forecast_value * adjustment)
        
        return adjusted_forecast

# Example: Adjust California forecast for outages
# (Requires outage service initialization - see eaglei_outages.py)
# analyzer = OutageImpactAnalyzer(outage_service)
# adjusted_forecast = analyzer.adjust_forecast_for_outages(
#     forecasts_48h, 
#     "California", 
#     datetime.now()
# )

# ======================================================================
# Code Block 8
# ======================================================================

"""Service for handling the massive EIA electricity parquet dataset."""

def __init__(self, parquet_path: str = "ELEC.parquet"):
    """Initialize EIA parquet service.
    
    Args:
        parquet_path: Path to the ELEC.parquet file.
    """
    self.parquet_path = parquet_path
    self.raw_data = None
    self._load_data()

def _load_data(self) -> None:
    """Load and parse the EIA parquet data."""
    try:
        logger.info(f"Loading EIA parquet data from {self.parquet_path}")
        self.raw_data = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(self.raw_data)} EIA records")
        
    except Exception as e:
        logger.error(f"Failed to load EIA parquet data: {e}")
        self.raw_data = pd.DataFrame()

def search_series(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search for series matching a query."""
    if self.raw_data is None or self.raw_data.empty:
        return []
    
    results = []
    column_name = self.raw_data.columns[0]

# ======================================================================
# Code Block 9
# ======================================================================

for i, row in self.raw_data.iterrows():
        if len(results) >= limit:
            break
            
        json_str = row[column_name]
        if query.lower() in json_str.lower():
            try:
                parsed = json.loads(json_str)
                results.append({
                    "series_id": parsed.get('series_id', ''),
                    "name": parsed.get('name', ''),
                    "units": parsed.get('units', ''),
                    "geography": parsed.get('geography', ''),
                    "start": parsed.get('start', ''),
                    "end": parsed.get('end', ''),
                    "data_points": len(parsed.get('data', []))
                })
            except json.JSONDecodeError:
                continue
    
    # return results

def get_time_series_data(self, series_id: str) -> Dict[str, Any]:
    """Get complete time series data for a specific series."""
    if self.raw_data is None or self.raw_data.empty:
        return {}
    
    column_name = self.raw_data.columns[0]
    
    for i, row in self.raw_data.iterrows():
        json_str = row[column_name]
        try:
            pass
        except Exception:
            pass
            parsed = json.loads(json_str)
            if parsed.get('series_id') == series_id:
                pass

# ======================================================================
# Code Block 10
# # ======================================================================

# data_points = parsed.get('data', [])
                
                dates = []
                values = []
                
                for point in data_points:
                    if len(point) >= 2:
                        date_str = point[0]
                        value = point[1]

# ======================================================================
# Code Block 11
# ======================================================================

if len(date_str) == 6:
                            year = int(date_str[:4])
                            month = int(date_str[4:])
                            dates.append(f"{year}-{month:02d}-01")
                            values.append(value)
                
                # return {
                    # "series_id": series_id,
                    # "name": parsed.get('name', ''),
                    # "units": parsed.get('units', ''),
                    # "dates": dates,
                    # "values": values,
                    # "metadata": {
                        # "lat": parsed.get('lat'),
                        # "lon": parsed.get('lon'),
                        # "geography": parsed.get('geography'),
                        # "start": parsed.get('start'),
                        # "end": parsed.get('end')
                    # }
                # }
                
        # except json.JSONDecodeError:
            # continue
    
    # return {}

# ======================================================================
# Code Block 12
# ======================================================================

print(f"Series: {result['series_id']}")
print(f"Name: {result['name']}")
print(f"Data points: {result['data_points']}")
print()

# ======================================================================
# Code Block 13
# ======================================================================

series_data = eia_service.get_time_series_data(results[0]['series_id'])
print(f"Retrieved {len(series_data['values'])} monthly data points")
print(f"Date range: {series_data['dates'][0]} to {series_data['dates'][-1]}")

# ======================================================================
# Code Block 14
# ======================================================================

"""Prepare feature dataset for modeling.

Args:
    load_data: DataFrame with columns [ts_utc, mw, ba]
    lookback_days: Number of days of historical data to use.
    
Returns:
    DataFrame with engineered features.
"""
df = load_data.copy()
df['ts_utc'] = pd.to_datetime(df['ts_utc'])
df = df.sort_values('ts_utc').reset_index(drop=True)

# ======================================================================
# Code Block 15
# ======================================================================

df['hour'] = df['ts_utc'].dt.hour
df['dow'] = df['ts_utc'].dt.dayofweek + 1  # 1=Monday
df['month'] = df['ts_utc'].dt.month
df['day_of_year'] = df['ts_utc'].dt.dayofyear
df['is_weekend'] = (df['dow'] >= 6).astype(int)

# ======================================================================
# Code Block 16
# ======================================================================

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# ======================================================================
# Code Block 17
# ======================================================================

df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

# ======================================================================
# Code Block 18
# ======================================================================

df['mw_lag1'] = df['mw'].shift(1)      # 1 hour ago
df['mw_lag24'] = df['mw'].shift(24)    # Same hour yesterday
df['mw_lag168'] = df['mw'].shift(168)  # Same hour last week

# ======================================================================
# Code Block 19
# ======================================================================

df['mw_ma24'] = df['mw'].rolling(window=24, min_periods=1).mean()
df['mw_ma168'] = df['mw'].rolling(window=168, min_periods=1).mean()
df['mw_std24'] = df['mw'].rolling(window=24, min_periods=1).std()

# ======================================================================
# Code Block 20
# ======================================================================

df['temperature'] = 70 + 20 * np.sin(2 * np.pi * df['day_of_year'] / 365.25) + \
                   10 * np.sin(2 * np.pi * df['hour'] / 24)
df['temp_squared'] = df['temperature'] ** 2
df['cooling_degree_days'] = np.maximum(df['temperature'] - 65, 0)
df['heating_degree_days'] = np.maximum(55 - df['temperature'], 0)

# ======================================================================
# Code Block 21
# ======================================================================

df['is_holiday'] = 0

return df

# ======================================================================
# Code Block 22
# ======================================================================

# 'ts_utc': pd.date_range('2024-01-01', periods=24*90, freq='H'),
# 'mw': np.random.normal(25000, 5000, 24*90),  # Synthetic load
# 'ba': 'CAL-ALL'

# ======================================================================
# Code Block 23
# ======================================================================

ts_data = df[['ts_utc', 'mw']].copy()
ts_data = ts_data.dropna().sort_values('ts_utc')

if len(ts_data) < 168:  # Need at least 1 week of data
    print(f"Insufficient data: {len(ts_data)} records")
    return None

# ======================================================================
# Code Block 24
# ======================================================================

ts_data.set_index('ts_utc', inplace=True)
ts_series = ts_data['mw']

with mlflow.start_run(run_name=f"arima_{ba}"):
    pass

# ======================================================================
# Code Block 25
# ======================================================================

print(f"Training auto_arima model for {ba}...")
    
    # model = auto_arima(
        # ts_series,
        # start_p=1, start_q=1,
        # max_p=3, max_q=3,
        # seasonal=True,
        # start_P=0, start_Q=0,
        # max_P=2, max_Q=2,
        # m=24,  # 24-hour seasonality
        # stepwise=True,
        # suppress_warnings=True,
        # error_action='ignore'
    # )

# ======================================================================
# Code Block 26
# ======================================================================

fitted_values = model.fittedvalues()
    # actual_values = ts_series[fitted_values.index]

# ======================================================================
# Code Block 27
# ======================================================================

mape = mean_absolute_percentage_error(actual_values, fitted_values)
    # mae = mean_absolute_error(actual_values, fitted_values)

# ======================================================================
# Code Block 28
# ======================================================================

model_info = mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=f"leap_{ba}_arima"
    )
    
    # print(f"ARIMA model trained: MAPE={mape:.4f}, Order={model.order}")
    # return model_info.model_uri

# ======================================================================
# Code Block 29
# ======================================================================

feature_cols = [
    'mw_lag1', 'mw_lag24', 'mw_lag168',
    'mw_ma24', 'mw_ma168', 'mw_std24',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'dow', 'month', 'is_weekend', 'is_holiday',
    'temperature', 'temp_squared',
    'cooling_degree_days', 'heating_degree_days'
]

# ======================================================================
# Code Block 30
# ======================================================================

model_df = df[feature_cols + ['mw']].dropna()

if len(model_df) < 168:
    print(f"Insufficient data: {len(model_df)} records")
    return None

X = model_df[feature_cols]
y = model_df['mw']

with mlflow.start_run(run_name=f"lightgbm_{ba}"):
    pass

# ======================================================================
# Code Block 31
# ======================================================================

tscv = TimeSeriesSplit(n_splits=5)
    
    # model = LGBMRegressor(
        # n_estimators=500,
        # learning_rate=0.1,
        # max_depth=8,
        # num_leaves=31,
        # random_state=42,
        # verbose=-1
    # )

# ======================================================================
# Code Block 32
# ======================================================================

cv_predictions = np.full(len(y), np.nan)
    
    # for train_idx, val_idx in tscv.split(X):
        # X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        # y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # model.fit(X_train, y_train)
        # cv_predictions[val_idx] = model.predict(X_val)

# ======================================================================
# Code Block 33
# ======================================================================

valid_mask = ~np.isnan(cv_predictions)
    # mape = mean_absolute_percentage_error(y[valid_mask], cv_predictions[valid_mask])
    # mae = mean_absolute_error(y[valid_mask], cv_predictions[valid_mask])
    # rmse = np.sqrt(mean_squared_error(y[valid_mask], cv_predictions[valid_mask]))

# ======================================================================
# Code Block 34
# ======================================================================

importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # mlflow.log_text(importance_df.to_string(), "feature_importance.txt")

# ======================================================================
# Code Block 35
# ======================================================================

model_info = mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=f"leap_{ba}_lightgbm"
    )
    
    # print(f"LightGBM trained: MAPE={mape:.4f}")
    # print(f"\nTop 5 features:")
    # print(importance_df.head())
    
    # return model_info.model_uri

# ======================================================================
# Code Block 36
# ======================================================================

"""Generate multi-hour forecast using trained model.

Args:
    model: Trained LightGBM model.
    df: Historical data with features.
    horizon_hours: Number of hours to forecast.
    
Returns:
    List of forecast values.
"""
feature_cols = [
    'mw_lag1', 'mw_lag24', 'mw_lag168',
    'mw_ma24', 'mw_ma168', 'mw_std24',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'dow', 'month', 'is_weekend', 'is_holiday',
    'temperature', 'temp_squared',
    'cooling_degree_days', 'heating_degree_days'
]

# ======================================================================
# Code Block 37
# ======================================================================

last_row = df.dropna(subset=feature_cols).iloc[-1].copy()
forecasts = []

for i in range(horizon_hours):
    pass

# ======================================================================
# Code Block 38
# ======================================================================

future_time = pd.to_datetime(last_row['ts_utc']) + timedelta(hours=i+1)
    # hour = future_time.hour
    # day_of_year = future_time.dayofyear
    
    # last_row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    # last_row['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    # last_row['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    # last_row['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    # last_row['dow'] = future_time.dayofweek + 1
    # last_row['month'] = future_time.month
    # last_row['is_weekend'] = int(last_row['dow'] >= 6)

# ======================================================================
# Code Block 39
# ======================================================================

last_row['temperature'] = 70 + 20 * np.sin(2 * np.pi * day_of_year / 365.25) + \
                             10 * np.sin(2 * np.pi * hour / 24)
    # last_row['temp_squared'] = last_row['temperature'] ** 2
    # last_row['cooling_degree_days'] = max(last_row['temperature'] - 65, 0)
    # last_row['heating_degree_days'] = max(55 - last_row['temperature'], 0)

# ======================================================================
# Code Block 40
# ======================================================================

X = last_row[feature_cols].values.reshape(1, -1)

# ======================================================================
# Code Block 41
# ======================================================================

forecast = model.predict(X)[0]
    # forecasts.append(forecast)

# ======================================================================
# Code Block 42
# ======================================================================

last_row['mw_lag1'] = forecast
    # last_row['mw_ma24'] = 0.95 * last_row['mw_ma24'] + 0.05 * forecast

return forecasts

# ======================================================================
# Code Block 43
# ======================================================================

print(f"Hour {i+1}: {forecast:,.0f} MW")

# ======================================================================
# Code Block 44
# ======================================================================

"""Apply scenario adjustments to feature DataFrame.

Args:
    df: Base features DataFrame.
    scenario_id: Scenario identifier.
    
Returns:
    Modified DataFrame with scenario adjustments.
"""
scenario_df = df.copy()

if scenario_id == "hot_weather":
    pass

# ======================================================================
# Code Block 45
# ======================================================================

scenario_df['temperature'] += 15
    # scenario_df['temp_squared'] = scenario_df['temperature'] ** 2
    # scenario_df['cooling_degree_days'] = np.maximum(scenario_df['temperature'] - 65, 0)
    # scenario_df['heating_degree_days'] = np.maximum(55 - scenario_df['temperature'], 0)

# elif scenario_id == "high_growth":

# ======================================================================
# Code Block 46
# ======================================================================

lag_cols = ['mw_lag1', 'mw_lag24', 'mw_lag168', 'mw_ma24', 'mw_ma168']
    # for col in lag_cols:
        # if col in scenario_df.columns:
            # scenario_df[col] *= 1.05

# elif scenario_id == "demand_response":

# ======================================================================
# Code Block 47
# ======================================================================

peak_hours = (scenario_df['hour'].between(16, 21))
    # scenario_df.loc[peak_hours, 'mw_lag1'] *= 0.9
    # scenario_df.loc[peak_hours, 'mw_lag24'] *= 0.9

# elif scenario_id == "major_outage":

# ======================================================================
# Code Block 48
# ======================================================================

scenario_df['mw_lag1'] *= 0.8
    # scenario_df['mw_ma24'] *= 0.85  # Partial recovery

return scenario_df

# ======================================================================
# Code Block 49
# ======================================================================

if scenario == "baseline":
    scenario_features = sample_features.copy()
else:
    scenario_features = apply_scenario(sample_features, scenario)

forecasts = generate_forecast(model, scenario_features, horizon_hours=24)
scenario_forecasts[scenario] = forecasts

# ======================================================================
# Code Block 50
# ======================================================================

peak_load = forecasts[17]  # Hour 18 (0-indexed)
print(f"{scenario:20s}: {peak_load:>8,.0f} MW")

# ======================================================================
# Code Block 51
# ======================================================================

scenario_peak = scenario_forecasts[scenario][17]
diff_pct = (scenario_peak / baseline_peak - 1) * 100
print(f"  {scenario} vs baseline: {diff_pct:+.1f}%")

# ======================================================================
# Code Block 52
# ======================================================================

"""Analyze how outages affect load forecasts."""

def __init__(self, outage_service):
    self.outage_service = outage_service

def get_outage_adjustment_factor(self, state: str, timestamp: datetime) -> float:
    """Calculate load adjustment factor based on current outages.
    
    Args:
        state: State code (e.g., 'Texas')
        timestamp: Current timestamp
        
    Returns:
        Adjustment factor (1.0 = no adjustment, 0.9 = 10% reduction)
    """

# ======================================================================
# Code Block 53
# ======================================================================

year = timestamp.year
    # outage_data = self.outage_service.get_outage_data(
        # year=year,
        # state=state,
        # start_date=timestamp.strftime("%Y-%m-%d"),
        # end_date=timestamp.strftime("%Y-%m-%d"),
        # limit=100
    # )
    
    # if not outage_data:
        # return 1.0

# ======================================================================
# Code Block 54
# ======================================================================

total_customers_out = sum(r['customers_out'] for r in outage_data)
    # total_customers = sum(r.get('total_customers', 0) for r in outage_data if r.get('total_customers'))
    
    # if total_customers == 0:
        # return 1.0
    
    # outage_rate = total_customers_out / total_customers

# ======================================================================
# Code Block 55
# ======================================================================

adjustment_factor = 1.0 - (outage_rate * 0.8)
    
    # return max(0.5, adjustment_factor)  # Cap at 50% reduction

def adjust_forecast_for_outages(self, forecast: List[float], state: str, 
                                base_time: datetime) -> List[float]:
    """Adjust forecast based on expected or ongoing outages.
    
    Args:
        forecast: Base forecast values
        state: State code
        base_time: Base timestamp for forecast
        
    Returns:
        Adjusted forecast values
    """
    adjusted_forecast = []
    
    for i, forecast_value in enumerate(forecast):
        forecast_time = base_time + timedelta(hours=i)
        adjustment = self.get_outage_adjustment_factor(state, forecast_time)
        adjusted_forecast.append(forecast_value * adjustment)
    
    return adjusted_forecast
