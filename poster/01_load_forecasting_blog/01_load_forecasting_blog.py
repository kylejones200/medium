#!/usr/bin/env python3
"""
Python code extracted from 01_load_forecasting_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_realistic_load_curve(base_load_mw=9300, date=None):
    """
    Generate 24-hour load curve with realistic intraday patterns.
    
    Parameters:
    -----------
    base_load_mw : float
        System peak capacity in megawatts
    date : datetime
        Reference date for the load curve
    
    Returns:
    --------
    pd.DataFrame : Hourly load data with load factor and pricing
    """
    if date is None:
        date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    hourly_data = []
    
    for hour in range(24):
        # Load factor varies throughout the day
        if 6 <= hour <= 9:  # Morning ramp
            load_factor = 0.85 + np.random.uniform(-0.03, 0.03)
        elif 17 <= hour <= 21:  # Evening peak
            load_factor = 0.95 + np.random.uniform(-0.03, 0.03)
        elif 22 <= hour <= 5:  # Night valley
            load_factor = 0.60 + np.random.uniform(-0.03, 0.03)
        else:  # Day hours
            load_factor = 0.75 + np.random.uniform(-0.03, 0.03)
        
        peak_load = base_load_mw * load_factor
        average_load = peak_load * 0.70
        
        # Price correlates with load
        base_price = 82.44  # $/MWh
        price_multiplier = 0.8 + (load_factor * 0.4)
        lmp_price = base_price * price_multiplier
        
        hourly_data.append({
            'timestamp': date + timedelta(hours=hour),
            'hour': hour,
            'peak_load_mw': peak_load,
            'average_load_mw': average_load,
            'load_factor': load_factor,
            'lmp_price': lmp_price
        })
    
    return pd.DataFrame(hourly_data)

# Generate load curve
load_data = generate_realistic_load_curve()
print(f"Daily Peak: {load_data['peak_load_mw'].max():.0f} MW")
print(f"Daily Valley: {load_data['peak_load_mw'].min():.0f} MW")
print(f"System Load Factor: {load_data['load_factor'].mean():.2%}")

# ======================================================================
# Code Block 2
# ======================================================================

def calculate_price_elasticity(load_data):
    """
    Calculate how price responds to load changes.
    
    Returns price-load elasticity and identifies high-margin trading hours.
    """
    # Calculate load change percentage
    load_data['load_pct_change'] = load_data['peak_load_mw'].pct_change() * 100
    load_data['price_pct_change'] = load_data['lmp_price'].pct_change() * 100
    
    # Identify high-value trading hours
    load_data['trading_opportunity'] = (
        (load_data['lmp_price'] > load_data['lmp_price'].quantile(0.75)) &
        (load_data['load_factor'] > 0.85)
    )
    
    # Calculate price elasticity
    elasticity = load_data['price_pct_change'].std() / load_data['load_pct_change'].std()
    
    high_value_hours = load_data[load_data['trading_opportunity']]
    
    return {
        'price_elasticity': elasticity,
        'high_value_hours': high_value_hours['hour'].tolist(),
        'peak_price': load_data['lmp_price'].max(),
        'valley_price': load_data['lmp_price'].min(),
        'price_spread': load_data['lmp_price'].max() - load_data['lmp_price'].min()
    }

elasticity_results = calculate_price_elasticity(load_data)
print(f"Price Elasticity: {elasticity_results['price_elasticity']:.2f}")
print(f"High-Value Trading Hours: {elasticity_results['high_value_hours']}")
print(f"Peak-Valley Spread: ${elasticity_results['price_spread']:.2f}/MWh")

# ======================================================================
# Code Block 3
# ======================================================================

def forecast_week_ahead(base_load_mw=9300, days=7):
    """
    Generate week-ahead load forecast incorporating daily patterns.
    
    Accounts for:
    - Weekday vs weekend patterns
    - Seasonal load variations
    - Weather-driven demand shifts
    """
    forecast_data = []
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        weekday = current_date.weekday()
        
        # Weekend load patterns differ from weekdays
        if weekday >= 5:  # Weekend
            daily_base = base_load_mw * 0.85
        else:  # Weekday
            daily_base = base_load_mw
        
        # Generate hourly data for this day
        daily_curve = generate_realistic_load_curve(daily_base, current_date)
        daily_curve['day_of_week'] = weekday
        daily_curve['day_type'] = 'Weekend' if weekday >= 5 else 'Weekday'
        
        forecast_data.append(daily_curve)
    
    # Combine all days
    full_forecast = pd.concat(forecast_data, ignore_index=True)
    
    # Calculate daily aggregates
    daily_summary = full_forecast.groupby(full_forecast['timestamp'].dt.date).agg({
        'peak_load_mw': 'max',
        'average_load_mw': 'mean',
        'lmp_price': ['mean', 'max'],
        'load_factor': 'mean'
    }).round(2)
    
    return full_forecast, daily_summary

week_forecast, week_summary = forecast_week_ahead()
print("Week-Ahead Load Forecast:")
print(week_summary)

# ======================================================================
# Code Block 4
# ======================================================================

def apply_weather_adjustment(load_data, temperature_f, humidity_pct):
    """
    Adjust load forecast based on weather conditions.
    
    Parameters:
    -----------
    load_data : pd.DataFrame
        Base load forecast
    temperature_f : float
        Expected temperature in Fahrenheit
    humidity_pct : float
        Expected humidity percentage
    
    Returns:
    --------
    pd.DataFrame : Weather-adjusted load forecast
    """
    # Temperature adjustment (simplified cooling degree day model)
    cooling_threshold = 75
    heating_threshold = 55
    
    if temperature_f > cooling_threshold:
        # Cooling load increases exponentially with temperature
        cooling_multiplier = 1 + ((temperature_f - cooling_threshold) * 0.025)
    elif temperature_f < heating_threshold:
        # Heating load increases with cold
        heating_multiplier = 1 + ((heating_threshold - temperature_f) * 0.015)
        cooling_multiplier = heating_multiplier
    else:
        cooling_multiplier = 1.0
    
    # Humidity adjustment (humidity increases perceived temperature)
    if humidity_pct > 60 and temperature_f > 75:
        humidity_multiplier = 1 + ((humidity_pct - 60) * 0.005)
    else:
        humidity_multiplier = 1.0
    
    # Apply weather adjustments
    weather_factor = cooling_multiplier * humidity_multiplier
    
    load_data_adjusted = load_data.copy()
    load_data_adjusted['peak_load_mw'] *= weather_factor
    load_data_adjusted['average_load_mw'] *= weather_factor
    load_data_adjusted['lmp_price'] *= (weather_factor ** 1.5)  # Price response is non-linear
    
    load_data_adjusted['weather_adjustment'] = weather_factor
    load_data_adjusted['temperature_f'] = temperature_f
    load_data_adjusted['humidity_pct'] = humidity_pct
    
    return load_data_adjusted

# Compare normal day vs hot summer day
normal_day = generate_realistic_load_curve()
hot_day = apply_weather_adjustment(normal_day.copy(), temperature_f=98, humidity_pct=75)

print(f"Normal Day Peak: {normal_day['peak_load_mw'].max():.0f} MW")
print(f"Hot Day Peak: {hot_day['peak_load_mw'].max():.0f} MW")
print(f"Peak Increase: {(hot_day['peak_load_mw'].max() / normal_day['peak_load_mw'].max() - 1) * 100:.1f}%")
print(f"\nNormal Day Peak Price: ${normal_day['lmp_price'].max():.2f}/MWh")
print(f"Hot Day Peak Price: ${hot_day['lmp_price'].max():.2f}/MWh")
print(f"Price Increase: {(hot_day['lmp_price'].max() / normal_day['lmp_price'].max() - 1) * 100:.1f}%")

# ======================================================================
# Code Block 5
# ======================================================================

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

def build_ml_load_forecast(historical_data, forecast_horizon=24):
    """
    Build machine learning model for load forecasting.
    
    Uses gradient boosting to capture non-linear relationships
    between features and load.
    """
    # Create feature matrix
    features = []
    targets = []
    
    for i in range(len(historical_data) - forecast_horizon):
        # Lag features
        lag_1 = historical_data['peak_load_mw'].iloc[i]
        lag_24 = historical_data['peak_load_mw'].iloc[max(0, i-24)]
        lag_168 = historical_data['peak_load_mw'].iloc[max(0, i-168)]  # Week ago
        
        # Time features
        hour = historical_data['hour'].iloc[i]
        day_of_week = historical_data['timestamp'].iloc[i].weekday()
        
        # Cyclical encoding of hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        feature_vec = [
            lag_1, lag_24, lag_168,
            hour_sin, hour_cos,
            day_of_week,
            historical_data['load_factor'].iloc[i]
        ]
        
        features.append(feature_vec)
        targets.append(historical_data['peak_load_mw'].iloc[i + forecast_horizon])
    
    X = np.array(features)
    y = np.array(targets)
    
    # Train-test split (80-20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train gradient boosting model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    mae = np.mean(np.abs(predictions - y_test))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'model': model,
        'scaler': scaler,
        'train_r2': train_score,
        'test_r2': test_score,
        'mae_mw': mae,
        'mape_pct': mape,
        'predictions': predictions,
        'actuals': y_test
    }

# Generate synthetic historical data for demonstration
historical_periods = 10
historical_data = []
for _ in range(historical_periods):
    period_data = generate_realistic_load_curve()
    historical_data.append(period_data)
historical_df = pd.concat(historical_data, ignore_index=True)

# Build and evaluate model
ml_results = build_ml_load_forecast(historical_df)
print(f"Model Performance:")
# print(f"Training R²: {ml_results['train_r2']:.3f}")
# print(f"Testing R²: {ml_results['test_r2']:.3f}")
print(f"Mean Absolute Error: {ml_results['mae_mw']:.2f} MW")
print(f"Mean Absolute Percentage Error: {ml_results['mape_pct']:.2f}%")

# ======================================================================
# Code Block 6
# ======================================================================

def update_forecast_realtime(base_forecast, actual_load_mw, current_hour):
    """
    Update remaining daily forecast based on actual load observed.
    
    Uses forecast error from current hour to adjust remaining hours.
    """
    # Calculate forecast error
    forecast_load = base_forecast[base_forecast['hour'] == current_hour]['peak_load_mw'].values[0]
    forecast_error_pct = (actual_load_mw - forecast_load) / forecast_load
    
    # Update remaining hours
    updated_forecast = base_forecast.copy()
    
    # Apply dampened error correction to future hours
    for i in range(current_hour + 1, 24):
        # Error impact decays with forecast horizon
        hours_ahead = i - current_hour
        dampening_factor = np.exp(-0.1 * hours_ahead)
        
        adjustment = forecast_error_pct * dampening_factor
        updated_forecast.loc[updated_forecast['hour'] == i, 'peak_load_mw'] *= (1 + adjustment)
        updated_forecast.loc[updated_forecast['hour'] == i, 'average_load_mw'] *= (1 + adjustment)
        
        # Update prices accordingly
        updated_forecast.loc[updated_forecast['hour'] == i, 'lmp_price'] *= (1 + adjustment * 1.2)
    
    return updated_forecast, forecast_error_pct

# Example: Update forecast at hour 10
base_forecast = generate_realistic_load_curve()
actual_load_observed = base_forecast[base_forecast['hour'] == 10]['peak_load_mw'].values[0] * 1.08  # 8% higher than forecast

updated_forecast, error_pct = update_forecast_realtime(base_forecast, actual_load_observed, 10)

print(f"Forecast Error at Hour 10: {error_pct:.2%}")
print(f"\nOriginal Peak Price (Hour 19): ${base_forecast[base_forecast['hour'] == 19]['lmp_price'].values[0]:.2f}/MWh")
print(f"Updated Peak Price (Hour 19): ${updated_forecast[updated_forecast['hour'] == 19]['lmp_price'].values[0]:.2f}/MWh")

# ======================================================================
# Code Block 7
# ======================================================================

"""
Generate 24-hour load curve with realistic intraday patterns.

Parameters:
-----------
base_load_mw : float
    System peak capacity in megawatts
date : datetime
    Reference date for the load curve

Returns:
--------
pd.DataFrame : Hourly load data with load factor and pricing
"""
if date is None:
    date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

hourly_data = []

for hour in range(24):
    pass

# ======================================================================
# Code Block 8
# ======================================================================

if 6 <= hour <= 9:  # Morning ramp
        load_factor = 0.85 + np.random.uniform(-0.03, 0.03)
elif 17 <= hour <= 21:  # Evening peak
        load_factor = 0.95 + np.random.uniform(-0.03, 0.03)
elif 22 <= hour <= 5:  # Night valley
        load_factor = 0.60 + np.random.uniform(-0.03, 0.03)
else:  # Day hours
        load_factor = 0.75 + np.random.uniform(-0.03, 0.03)
    
peak_load = base_load_mw * load_factor
average_load = peak_load * 0.70

# ======================================================================
# Code Block 9
# ======================================================================

base_price = 82.44  # $/MWh
price_multiplier = 0.8 + (load_factor * 0.4)
lmp_price = base_price * price_multiplier
    
hourly_data.append({
        'timestamp': date + timedelta(hours=hour),
        'hour': hour,
        'peak_load_mw': peak_load,
        'average_load_mw': average_load,
        'load_factor': load_factor,
        'lmp_price': lmp_price
    })

return pd.DataFrame(hourly_data)

# ======================================================================
# Code Block 10
# ======================================================================

load_data['load_pct_change'] = load_data['peak_load_mw'].pct_change() * 100
load_data['price_pct_change'] = load_data['lmp_price'].pct_change() * 100

# ======================================================================
# Code Block 11
# ======================================================================

load_data['trading_opportunity'] = (
    (load_data['lmp_price'] > load_data['lmp_price'].quantile(0.75)) &
    (load_data['load_factor'] > 0.85)
)

# ======================================================================
# Code Block 12
# ======================================================================

elasticity = load_data['price_pct_change'].std() / load_data['load_pct_change'].std()

high_value_hours = load_data[load_data['trading_opportunity']]

return {
    'price_elasticity': elasticity,
    'high_value_hours': high_value_hours['hour'].tolist(),
    'peak_price': load_data['lmp_price'].max(),
    'valley_price': load_data['lmp_price'].min(),
    'price_spread': load_data['lmp_price'].max() - load_data['lmp_price'].min()
}

# ======================================================================
# Code Block 13
# ======================================================================

"""
Generate week-ahead load forecast incorporating daily patterns.

Accounts for:
- Weekday vs weekend patterns
- Seasonal load variations
- Weather-driven demand shifts
"""
forecast_data = []
start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

for day in range(days):
    current_date = start_date + timedelta(days=day)
    weekday = current_date.weekday()

# ======================================================================
# Code Block 14
# ======================================================================

if weekday >= 5:  # Weekend
        daily_base = base_load_mw * 0.85
else:  # Weekday
        daily_base = base_load_mw

# ======================================================================
# Code Block 15
# ======================================================================

daily_curve = generate_realistic_load_curve(daily_base, current_date)
daily_curve['day_of_week'] = weekday
daily_curve['day_type'] = 'Weekend' if weekday >= 5 else 'Weekday'
    
forecast_data.append(daily_curve)

# ======================================================================
# Code Block 16
# ======================================================================

full_forecast = pd.concat(forecast_data, ignore_index=True)

# ======================================================================
# Code Block 17
# ======================================================================

daily_summary = full_forecast.groupby(full_forecast['timestamp'].dt.date).agg({
    'peak_load_mw': 'max',
    'average_load_mw': 'mean',
    'lmp_price': ['mean', 'max'],
    'load_factor': 'mean'
}).round(2)

return full_forecast, daily_summary

# ======================================================================
# Code Block 18
# ======================================================================

cooling_threshold = 75
heating_threshold = 55

if temperature_f > cooling_threshold:
    pass

# ======================================================================
# Code Block 19
# ======================================================================

cooling_multiplier = 1 + ((temperature_f - cooling_threshold) * 0.025)
# elif temperature_f < heating_threshold:

# ======================================================================
# Code Block 20
# ======================================================================

heating_multiplier = 1 + ((heating_threshold - temperature_f) * 0.015)
cooling_multiplier = heating_multiplier
# else:
cooling_multiplier = 1.0

# ======================================================================
# Code Block 21
# ======================================================================

if humidity_pct > 60 and temperature_f > 75:
    humidity_multiplier = 1 + ((humidity_pct - 60) * 0.005)
else:
    humidity_multiplier = 1.0

# ======================================================================
# Code Block 22
# ======================================================================

weather_factor = cooling_multiplier * humidity_multiplier

load_data_adjusted = load_data.copy()
load_data_adjusted['peak_load_mw'] *= weather_factor
load_data_adjusted['average_load_mw'] *= weather_factor
load_data_adjusted['lmp_price'] *= (weather_factor ** 1.5)  # Price response is non-linear

load_data_adjusted['weather_adjustment'] = weather_factor
load_data_adjusted['temperature_f'] = temperature_f
load_data_adjusted['humidity_pct'] = humidity_pct

return load_data_adjusted

# ======================================================================
# Code Block 23
# ======================================================================

features = []
targets = []

for i in range(len(historical_data) - forecast_horizon):
    pass

# ======================================================================
# Code Block 24
# ======================================================================

lag_1 = historical_data['peak_load_mw'].iloc[i]
lag_24 = historical_data['peak_load_mw'].iloc[max(0, i-24)]
lag_168 = historical_data['peak_load_mw'].iloc[max(0, i-168)]  # Week ago

# ======================================================================
# Code Block 25
# ======================================================================

hour = historical_data['hour'].iloc[i]
day_of_week = historical_data['timestamp'].iloc[i].weekday()

# ======================================================================
# Code Block 26
# ======================================================================

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
    
feature_vec = [
        lag_1, lag_24, lag_168,
        hour_sin, hour_cos,
        day_of_week,
        historical_data['load_factor'].iloc[i]
    ]
    
features.append(feature_vec)
targets.append(historical_data['peak_load_mw'].iloc[i + forecast_horizon])

X = np.array(features)
y = np.array(targets)

# ======================================================================
# Code Block 27
# ======================================================================

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ======================================================================
# Code Block 28
# ======================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================================================
# Code Block 29
# ======================================================================

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ======================================================================
# Code Block 30
# ======================================================================

train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

# ======================================================================
# Code Block 31
# ======================================================================

predictions = model.predict(X_test_scaled)
mae = np.mean(np.abs(predictions - y_test))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

return {
    'model': model,
    'scaler': scaler,
    'train_r2': train_score,
    'test_r2': test_score,
    'mae_mw': mae,
    'mape_pct': mape,
    'predictions': predictions,
    'actuals': y_test
}

# ======================================================================
# Code Block 32
# ======================================================================

period_data = generate_realistic_load_curve()
historical_data.append(period_data)

# ======================================================================
# Code Block 33
# ======================================================================

forecast_load = base_forecast[base_forecast['hour'] == current_hour]['peak_load_mw'].values[0]
forecast_error_pct = (actual_load_mw - forecast_load) / forecast_load

# ======================================================================
# Code Block 34
# ======================================================================

updated_forecast = base_forecast.copy()

# ======================================================================
# Code Block 35
# ======================================================================

hours_ahead = i - current_hour
dampening_factor = np.exp(-0.1 * hours_ahead)
    
adjustment = forecast_error_pct * dampening_factor
updated_forecast.loc[updated_forecast['hour'] == i, 'peak_load_mw'] *= (1 + adjustment)
updated_forecast.loc[updated_forecast['hour'] == i, 'average_load_mw'] *= (1 + adjustment)

# ======================================================================
# Code Block 36
# ======================================================================

updated_forecast.loc[updated_forecast['hour'] == i, 'lmp_price'] *= (1 + adjustment * 1.2)

return updated_forecast, forecast_error_pct
