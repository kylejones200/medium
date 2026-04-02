#!/usr/bin/env python3
"""
Python code extracted from 06_thermal_anomaly_detection_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

def fetch_modis_lst_data(latitude, longitude, start_date, end_date):
    """
    Fetch MODIS Land Surface Temperature data for a specific location.
    
    Uses NASA EARTHDATA AppEEARS API to retrieve MOD11A2 (8-day composite)
    data at 1km resolution.
    
    Parameters:
    -----------
    latitude : float
        Latitude of mine site center
    longitude : float
        Longitude of mine site center
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame : Time series of land surface temperature
    """
    # MODIS LST comes in Kelvin with scale factor 0.02
    # This generates realistic synthetic data matching MODIS characteristics
    # In production, use NASA EARTHDATA AppEEARS API
    
    dates = pd.date_range(start=start_date, end=end_date, freq='8D')
    
    # Baseline temperature (Kelvin) varies by season
    base_temp_k = 295  # ~22°C
    
    temperatures = []
    for date in dates:
        # Seasonal variation
        day_of_year = date.timetuple().tm_yday
        seasonal = 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Random weather variation
        weather_noise = np.random.normal(0, 3)
        
        # Base temperature
        temp_k = base_temp_k + seasonal + weather_noise
        
        temperatures.append({
            'date': date,
            'lst_day_kelvin': temp_k,
            'lst_night_kelvin': temp_k - 12,  # Night is cooler
            'quality_flag': 0,  # 0 = good quality
            'latitude': latitude,
            'longitude': longitude
        })
    
    return pd.DataFrame(temperatures)

def kelvin_to_celsius(kelvin):
    """Convert Kelvin to Celsius."""
    return kelvin - 273.15

# Example: Fetch data for a mine site in Western Australia
mine_lat, mine_lon = -30.5, 121.5
thermal_data = fetch_modis_lst_data(mine_lat, mine_lon, '2022-01-01', '2024-01-01')

# Convert to Celsius
thermal_data['lst_day_celsius'] = kelvin_to_celsius(thermal_data['lst_day_kelvin'])
thermal_data['lst_night_celsius'] = kelvin_to_celsius(thermal_data['lst_night_kelvin'])

print(f"Collected {len(thermal_data)} observations")
print(f"Temperature range: {thermal_data['lst_day_celsius'].min():.1f}°C to {thermal_data['lst_day_celsius'].max():.1f}°C")
print(f"Mean temperature: {thermal_data['lst_day_celsius'].mean():.1f}°C")

# Output:
# Collected 92 observations
# Temperature range: 6.3°C to 32.8°C
# Mean temperature: 21.1°C

# ======================================================================
# Code Block 2
# ======================================================================

def calculate_thermal_baseline(thermal_data, baseline_period_days=730):
    """
    Calculate statistical baseline for thermal data.
    
    Uses multi-year statistics to establish normal temperature
    ranges accounting for seasonal variation.
    
    Parameters:
    -----------
    thermal_data : pd.DataFrame
        Time series thermal data
    baseline_period_days : int
        Number of days to use for baseline (default 2 years)
    
    Returns:
    --------
    dict : Baseline statistics including seasonal patterns
    """
    # Calculate day-of-year patterns
    thermal_data['day_of_year'] = thermal_data['date'].dt.dayofyear
    
    # Group by week of year (52 weeks) for seasonal baseline
    thermal_data['week_of_year'] = thermal_data['date'].dt.isocalendar().week
    
    seasonal_baseline = thermal_data.groupby('week_of_year').agg({
        'lst_day_celsius': ['mean', 'std'],
        'lst_night_celsius': ['mean', 'std']
    }).reset_index()
    
    seasonal_baseline.columns = ['week', 'day_mean', 'day_std', 'night_mean', 'night_std']
    
    # Overall statistics
    overall_stats = {
        'day_mean': thermal_data['lst_day_celsius'].mean(),
        'day_std': thermal_data['lst_day_celsius'].std(),
        'day_p95': thermal_data['lst_day_celsius'].quantile(0.95),
        'day_p99': thermal_data['lst_day_celsius'].quantile(0.99),
        'night_mean': thermal_data['lst_night_celsius'].mean(),
        'night_std': thermal_data['lst_night_celsius'].std()
    }
    
    return {
        'seasonal': seasonal_baseline,
        'overall': overall_stats
    }

# Calculate baseline
baseline = calculate_thermal_baseline(thermal_data)

print("Thermal Baseline Statistics:")
print(f"  Day Mean: {baseline['overall']['day_mean']:.2f}°C")
print(f"  Day Std Dev: {baseline['overall']['day_std']:.2f}°C")
print(f"  Day 95th Percentile: {baseline['overall']['day_p95']:.2f}°C")
print(f"  Day 99th Percentile: {baseline['overall']['day_p99']:.2f}°C")
print(f"\nSeasonal variation range: {baseline['seasonal']['day_mean'].max() - baseline['seasonal']['day_mean'].min():.2f}°C")

# Output:
# Thermal Baseline Statistics:
#   Day Mean: 21.12°C
#   Day Std Dev: 6.52°C
#   Day 95th Percentile: 31.19°C
#   Day 99th Percentile: 32.18°C
# 
# Seasonal variation range: 21.01°C

# ======================================================================
# Code Block 3
# ======================================================================

def detect_thermal_anomalies(thermal_data, baseline, threshold_sigma=2.5):
    """
    Detect thermal anomalies using statistical thresholds.
    
    Compares observed temperatures against seasonal baselines
    to identify statistically significant deviations.
    
    Parameters:
    -----------
    thermal_data : pd.DataFrame
        Current thermal observations
    baseline : dict
        Baseline statistics from calculate_thermal_baseline
    threshold_sigma : float
        Number of standard deviations for anomaly threshold
    
    Returns:
    --------
    pd.DataFrame : Thermal data with anomaly flags and scores
    """
    result = thermal_data.copy()
    result['week_of_year'] = result['date'].dt.isocalendar().week
    
    # Merge with seasonal baseline
    result = result.merge(
        baseline['seasonal'], 
        left_on='week_of_year', 
        right_on='week', 
        how='left'
    )
    
    # Calculate z-scores (standard deviations from seasonal mean)
    result['day_z_score'] = (
        (result['lst_day_celsius'] - result['day_mean']) / result['day_std']
    )
    result['night_z_score'] = (
        (result['lst_night_celsius'] - result['night_mean']) / result['night_std']
    )
    
    # Flag anomalies
    result['day_anomaly'] = result['day_z_score'] > threshold_sigma
    result['night_anomaly'] = result['night_z_score'] > threshold_sigma
    result['any_anomaly'] = result['day_anomaly'] | result['night_anomaly']
    
    # Calculate anomaly severity score (0-100 scale)
    result['anomaly_score'] = np.clip(
        result['day_z_score'] * 20,  # 5 sigma = 100 points
        0, 
        100
    )
    
    return result

# Simulate anomaly by injecting elevated temperatures
thermal_data_with_anomaly = thermal_data.copy()
# Add anomaly in last 90 days
recent_dates = thermal_data_with_anomaly['date'] > (thermal_data_with_anomaly['date'].max() - timedelta(days=90))
thermal_data_with_anomaly.loc[recent_dates, 'lst_day_celsius'] += 8  # Add 8°C anomaly

# Detect anomalies
anomalies = detect_thermal_anomalies(thermal_data_with_anomaly, baseline)

# Summary
anomaly_count = anomalies['any_anomaly'].sum()
anomaly_pct = (anomaly_count / len(anomalies)) * 100
max_severity = anomalies['anomaly_score'].max()

print(f"Anomaly Detection Results:")
print(f"  Total Observations: {len(anomalies)}")
print(f"  Anomalies Detected: {anomaly_count} ({anomaly_pct:.1f}%)")
print(f"  Maximum Severity Score: {max_severity:.1f}/100")
if anomaly_count > 0:
    print(f"  Mean Temperature Deviation: {anomalies[anomalies['any_anomaly']]['day_z_score'].mean():.2f} sigma")

# Output:
# Anomaly Detection Results:
#   Total Observations: 92
#   Anomalies Detected: 9 (9.8%)
#   Maximum Severity Score: 100.0/100
#   Mean Temperature Deviation: 5.97 sigma

# ======================================================================
# Code Block 4
# ======================================================================

def analyze_mine_site_thermal(site_name, features_list):
    """
    Analyze thermal patterns across multiple mine features.
    
    Compares thermal behavior across tailings dams, waste dumps,
    processing facilities, and pit areas to identify relative risks.
    
    Parameters:
    -----------
    site_name : str
        Mine site identifier
    features_list : list of dict
        List of features with coordinates and descriptions
    
    Returns:
    --------
    pd.DataFrame : Comparative thermal analysis across features
    """
    all_results = []
    np.random.seed(42)  # For reproducibility
    
    # Feature-specific thermal adjustments (Pythonic dictionary lookup)
    thermal_adjustments = {
        'waste_dump': lambda n: np.random.normal(3, 1, n),
        'tailings_dam': lambda n: np.random.uniform(12, 18, n),
        'facility': lambda n: np.zeros(n),
        'pit': lambda n: np.zeros(n)
    }
    
    for feature in features_list:
        # Fetch thermal data for each feature
        thermal = fetch_modis_lst_data(
            feature['lat'], 
            feature['lon'],
            '2022-01-01', 
            '2024-01-01'
        )
        thermal['lst_day_celsius'] = kelvin_to_celsius(thermal['lst_day_kelvin'])
        thermal['lst_night_celsius'] = kelvin_to_celsius(thermal['lst_night_kelvin'])
        
        # Apply feature-specific thermal stress (Pythonic)
        feature_type = feature['type']
        recent = thermal['date'] > '2023-10-01'
        adjustment_func = thermal_adjustments.get(feature_type, lambda n: np.zeros(n))
        thermal.loc[recent, 'lst_day_celsius'] += adjustment_func(recent.sum())
        
        # Calculate baseline and detect anomalies
        baseline = calculate_thermal_baseline(thermal)
        anomalies = detect_thermal_anomalies(thermal, baseline)
        
        # Recent period analysis (last 90 days)
        recent_period = anomalies['date'] > (anomalies['date'].max() - timedelta(days=90))
        recent_anomalies = anomalies[recent_period]
        
        # Risk level calculation (Pythonic)
        max_score = recent_anomalies['anomaly_score'].max()
        risk_level = pd.cut([max_score], bins=[-np.inf, 40, 60, np.inf], 
                           labels=['LOW', 'MEDIUM', 'HIGH'])[0]
        
        feature_summary = {
            'site': site_name,
            'feature_name': feature['name'],
            'feature_type': feature['type'],
            'latitude': feature['lat'],
            'longitude': feature['lon'],
            'recent_mean_temp': recent_anomalies['lst_day_celsius'].mean(),
            'recent_max_temp': recent_anomalies['lst_day_celsius'].max(),
            'anomaly_count_90d': int(recent_anomalies['any_anomaly'].sum()),
            'max_anomaly_score': max_score,
            'mean_z_score': recent_anomalies['day_z_score'].mean(),
            'risk_level': risk_level
        }
        
        all_results.append(feature_summary)
    
    return pd.DataFrame(all_results)

# Define mine site features
mine_features = [
    {'name': 'Main Tailings Dam', 'type': 'tailings_dam', 'lat': -30.50, 'lon': 121.50},
    {'name': 'North Waste Dump', 'type': 'waste_dump', 'lat': -30.48, 'lon': 121.52},
    {'name': 'South Waste Dump', 'type': 'waste_dump', 'lat': -30.52, 'lon': 121.48},
    {'name': 'Processing Plant', 'type': 'facility', 'lat': -30.49, 'lon': 121.51},
    {'name': 'Open Pit', 'type': 'pit', 'lat': -30.51, 'lon': 121.49}
]

# Analyze all features
site_analysis = analyze_mine_site_thermal('Golden Grove Mine', mine_features)

print("\nMine Site Thermal Analysis Summary:")
print("=" * 80)
for _, row in site_analysis.sort_values('max_anomaly_score', ascending=False).iterrows():
    print(f"{row['feature_name']} ({row['feature_type']}):")
    print(f"  Risk Level: {row['risk_level']}")
    print(f"  Recent Max Temp: {row['recent_max_temp']:.1f}°C")
    print(f"  Anomalies (90d): {row['anomaly_count_90d']}")
    print(f"  Max Anomaly Score: {row['max_anomaly_score']:.1f}/100")
    print()

# Output:
# Mine Site Thermal Analysis Summary:
# ================================================================================
# Main Tailings Dam (tailings_dam):
#   Risk Level: LOW
#   Recent Max Temp: 37.4°C
#   Anomalies (90d): 0
#   Max Anomaly Score: 22.7/100
# 
# South Waste Dump (waste_dump):
#   Risk Level: LOW
#   Recent Max Temp: 24.4°C
#   Anomalies (90d): 0
#   Max Anomaly Score: 19.8/100
# 
# North Waste Dump (waste_dump):
#   Risk Level: LOW
#   Recent Max Temp: 23.8°C
#   Anomalies (90d): 0
#   Max Anomaly Score: 14.1/100

# ======================================================================
# Code Block 5
# ======================================================================

def analyze_thermal_trends(thermal_data, baseline, window_days=90):
    """
    Analyze thermal trends to identify developing problems.
    
    Uses moving averages and trend analysis to distinguish
    transient spikes from sustained thermal elevation.
    
    Parameters:
    -----------
    thermal_data : pd.DataFrame
        Thermal observations with dates
    baseline : dict
        Baseline statistics
    window_days : int
        Rolling window for trend analysis
    
    Returns:
    --------
    dict : Trend analysis results
    """
    # Calculate rolling statistics
    thermal_sorted = thermal_data.sort_values('date').copy()
    thermal_sorted['rolling_mean'] = thermal_sorted['lst_day_celsius'].rolling(
        window=window_days // 8, min_periods=3  # 8-day composites
    ).mean()
    thermal_sorted['rolling_max'] = thermal_sorted['lst_day_celsius'].rolling(
        window=window_days // 8, min_periods=3
    ).max()
    
    # Calculate trend over last 6 months
    recent_6mo = thermal_sorted[thermal_sorted['date'] > (thermal_sorted['date'].max() - timedelta(days=180))]

    # Calculate annual trend (Pythonic with conditional expression)
    annual_trend = (
        np.polyfit(np.arange(len(recent_6mo)), recent_6mo['lst_day_celsius'].values, 1)[0] * (365 / 8)
        if len(recent_6mo) >= 10
        else 0
    )

    # Compare recent period to baseline
    recent_30d = thermal_sorted[thermal_sorted['date'] > (thermal_sorted['date'].max() - timedelta(days=30))]
    recent_mean = recent_30d['lst_day_celsius'].mean()
    baseline_mean = baseline['overall']['day_mean']
    deviation_from_baseline = recent_mean - baseline_mean

    # Trend classification (Pythonic)
    trend_bins = [-np.inf, -0.5, 0.5, 2, np.inf]
    trend_labels = ['COOLING', 'STABLE', 'SLIGHT_WARMING', 'WARMING']
    
    # Use pd.cut for trend status
    trend_category = pd.cut([annual_trend], bins=trend_bins, labels=trend_labels)[0]
    
    # Map trend to urgency (allows duplicate values)
    urgency_map = {'COOLING': 'LOW', 'STABLE': 'LOW', 'SLIGHT_WARMING': 'MEDIUM', 'WARMING': 'HIGH'}
    urgency_category = urgency_map[trend_category]

    return {
        'annual_trend_celsius': annual_trend,
        'trend_status': trend_category,
        'urgency': urgency_category,
        'recent_mean': recent_mean,
        'baseline_mean': baseline_mean,
        'deviation': deviation_from_baseline,
        'recent_observations': len(recent_30d)
    }

# Analyze trend for tailings dam (with injected warming trend)
tailings_thermal = fetch_modis_lst_data(-30.50, 121.50, '2022-01-01', '2024-01-01')
tailings_thermal['lst_day_celsius'] = kelvin_to_celsius(tailings_thermal['lst_day_kelvin'])
tailings_thermal['lst_night_celsius'] = kelvin_to_celsius(tailings_thermal['lst_night_kelvin'])

# Inject warming trend in recent data
recent_mask = tailings_thermal['date'] > '2023-06-01'
days_recent = (tailings_thermal.loc[recent_mask, 'date'] - tailings_thermal.loc[recent_mask, 'date'].min()).dt.days
tailings_thermal.loc[recent_mask, 'lst_day_celsius'] += (days_recent / 180) * 6  # 6°C warming over 6 months

baseline_tailings = calculate_thermal_baseline(tailings_thermal)
trend_analysis = analyze_thermal_trends(tailings_thermal, baseline_tailings)

print("Thermal Trend Analysis:")
print("=" * 60)
print(f"Trend Status: {trend_analysis['trend_status']}")
print(f"Urgency Level: {trend_analysis['urgency']}")
print(f"Annual Trend: {trend_analysis['annual_trend_celsius']:+.2f}°C/year")
print(f"Recent Mean: {trend_analysis['recent_mean']:.2f}°C")
print(f"Baseline Mean: {trend_analysis['baseline_mean']:.2f}°C")
print(f"Deviation: {trend_analysis['deviation']:+.2f}°C")

# Output:
# Thermal Trend Analysis:
# ============================================================
# Trend Status: COOLING
# Urgency Level: LOW
# Annual Trend: -18.05°C/year
# Recent Mean: 22.98°C
# Baseline Mean: 22.44°C
# Deviation: +0.54°C

# ======================================================================
# Code Block 6
# ======================================================================

def correlate_satellite_ground(satellite_thermal, ground_measurements):
    """
    Correlate satellite thermal observations with ground-based sensors.
    
    Validates satellite anomalies against instrumentation and
    quantifies detection sensitivity.
    
    Parameters:
    -----------
    satellite_thermal : pd.DataFrame
        MODIS thermal observations
    ground_measurements : pd.DataFrame
        Ground-based temperature sensor data
    
    Returns:
    --------
    dict : Correlation metrics and validation statistics
    """
    # Merge datasets by date (within 4 days for 8-day MODIS composites)
    merged = pd.merge_asof(
        satellite_thermal[['date', 'lst_day_celsius']].sort_values('date'),
        ground_measurements[['date', 'ground_temp_celsius']].sort_values('date'),
        on='date',
        direction='nearest',
        tolerance=pd.Timedelta(days=4)
    )
    
    # Handle NaN and check for sufficient data (Pythonic)
    merged = merged.dropna()
    if len(merged) < 10:
        return {'error': f'Insufficient matched observations: {len(merged)}'}

    # Calculate correlation
    correlation = merged['lst_day_celsius'].corr(merged['ground_temp_celsius'])

    # Calculate bias (satellite - ground)
    merged['bias'] = merged['lst_day_celsius'] - merged['ground_temp_celsius']
    mean_bias = merged['bias'].mean()
    rmse = np.sqrt(np.mean(merged['bias'] ** 2))

    # Anomaly detection comparison (Pythonic)
    ground_baseline = ground_measurements['ground_temp_celsius'].mean()
    ground_std = ground_measurements['ground_temp_celsius'].std()
    sat_threshold = satellite_thermal['lst_day_celsius'].mean() + 2.5 * satellite_thermal['lst_day_celsius'].std()

    merged['ground_anomaly'] = (merged['ground_temp_celsius'] - ground_baseline) / ground_std > 2.5
    merged['satellite_anomaly'] = merged['lst_day_celsius'] > sat_threshold

    # Confusion matrix (Pythonic with pandas crosstab)
    confusion = pd.crosstab(merged['satellite_anomaly'], merged['ground_anomaly'])
    
    # Safe indexing with .get() for missing categories
    true_positives = confusion.get((True, True), 0)
    false_positives = confusion.get((True, False), 0)
    true_negatives = confusion.get((False, False), 0)
    false_negatives = confusion.get((False, True), 0)

    # Calculate metrics with safe division (Pythonic)
    sensitivity = true_positives / max(1, true_positives + false_negatives)
    specificity = true_negatives / max(1, true_negatives + false_positives)

    return {
        'correlation': correlation,
        'mean_bias': mean_bias,
        'rmse': rmse,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'matched_observations': len(merged)
    }

# Simulate ground sensor data
ground_data = thermal_data.copy()
ground_data['ground_temp_celsius'] = ground_data['lst_day_celsius'] + np.random.normal(2, 1.5, len(ground_data))

# Correlate
validation = correlate_satellite_ground(thermal_data, ground_data)

print("\nSatellite-Ground Correlation:")
print("=" * 60)

# Pythonic error handling with dict.get()
if 'error' in validation:
    print(f"Error: {validation['error']}")
else:
    print(f"Correlation Coefficient: {validation['correlation']:.3f}")
    print(f"Mean Bias: {validation['mean_bias']:+.2f}°C")
    print(f"RMSE: {validation['rmse']:.2f}°C")
    print(f"Anomaly Detection Sensitivity: {validation['sensitivity']:.1%}")
    print(f"Anomaly Detection Specificity: {validation['specificity']:.1%}")
    print(f"Matched Observations: {validation['matched_observations']}")

# Output:
# Satellite-Ground Correlation:
# ============================================================
# Correlation Coefficient: 0.974
# Mean Bias: -1.97°C
# RMSE: 2.54°C
# Anomaly Detection Sensitivity: 0.0%
# Anomaly Detection Specificity: 0.0%
# Matched Observations: 92

# ======================================================================
# Code Block 7
# ======================================================================

dates = pd.date_range(start=start_date, end=end_date, freq='8D')

# ======================================================================
# Code Block 8
# ======================================================================

base_temp_k = 295  # ~22°C

temperatures = []
for date in dates:

# ======================================================================
# Code Block 9
# ======================================================================

    pass
day_of_year = date.timetuple().tm_yday
seasonal = 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

# ======================================================================
# Code Block 10
# ======================================================================

weather_noise = np.random.normal(0, 3)

# ======================================================================
# Code Block 11
# ======================================================================

temp_k = base_temp_k + seasonal + weather_noise
    
temperatures.append({
        'date': date,
        'lst_day_kelvin': temp_k,
        'lst_night_kelvin': temp_k - 12,  # Night is cooler
        'quality_flag': 0,  # 0 = good quality
        'latitude': latitude,
        'longitude': longitude
    })

return pd.DataFrame(temperatures)

# ======================================================================
# Code Block 12
# ======================================================================

"""Convert Kelvin to Celsius."""
return kelvin - 273.15

# ======================================================================
# Code Block 13
# ======================================================================

thermal_data['day_of_year'] = thermal_data['date'].dt.dayofyear

# ======================================================================
# Code Block 14
# ======================================================================

thermal_data['week_of_year'] = thermal_data['date'].dt.isocalendar().week

seasonal_baseline = thermal_data.groupby('week_of_year').agg({
    'lst_day_celsius': ['mean', 'std'],
    'lst_night_celsius': ['mean', 'std']
}).reset_index()

seasonal_baseline.columns = ['week', 'day_mean', 'day_std', 'night_mean', 'night_std']

# ======================================================================
# Code Block 15
# ======================================================================

overall_stats = {
    'day_mean': thermal_data['lst_day_celsius'].mean(),
    'day_std': thermal_data['lst_day_celsius'].std(),
    'day_p95': thermal_data['lst_day_celsius'].quantile(0.95),
    'day_p99': thermal_data['lst_day_celsius'].quantile(0.99),
    'night_mean': thermal_data['lst_night_celsius'].mean(),
    'night_std': thermal_data['lst_night_celsius'].std()
}

return {
    'seasonal': seasonal_baseline,
    'overall': overall_stats
}

# ======================================================================
# Code Block 16
# ======================================================================

"""
Detect thermal anomalies using statistical thresholds.

Compares observed temperatures against seasonal baselines
to identify statistically significant deviations.

Parameters:
-----------
thermal_data : pd.DataFrame
    Current thermal observations
baseline : dict
    Baseline statistics from calculate_thermal_baseline
threshold_sigma : float
    Number of standard deviations for anomaly threshold

Returns:
--------
pd.DataFrame : Thermal data with anomaly flags and scores
"""
result = thermal_data.copy()
result['week_of_year'] = result['date'].dt.isocalendar().week

# ======================================================================
# Code Block 17
# ======================================================================

result = result.merge(
    baseline['seasonal'], 
    left_on='week_of_year', 
    right_on='week', 
    how='left'
)

# ======================================================================
# Code Block 18
# ======================================================================

result['day_z_score'] = (
    (result['lst_day_celsius'] - result['day_mean']) / result['day_std']
)
result['night_z_score'] = (
    (result['lst_night_celsius'] - result['night_mean']) / result['night_std']
)

# ======================================================================
# Code Block 19
# ======================================================================

result['day_anomaly'] = result['day_z_score'] > threshold_sigma
result['night_anomaly'] = result['night_z_score'] > threshold_sigma
result['any_anomaly'] = result['day_anomaly'] | result['night_anomaly']

# ======================================================================
# Code Block 20
# ======================================================================

result['anomaly_score'] = np.clip(
    result['day_z_score'] * 20,  # 5 sigma = 100 points
    0, 
    100
)

return result

# ======================================================================
# Code Block 21
# ======================================================================

print(f"  Mean Temperature Deviation: {anomalies[anomalies['any_anomaly']]['day_z_score'].mean():.2f} sigma")

# ======================================================================
# Code Block 22
# ======================================================================

"""
Analyze thermal patterns across multiple mine features.

Compares thermal behavior across tailings dams, waste dumps,
processing facilities, and pit areas to identify relative risks.

Parameters:
-----------
site_name : str
    Mine site identifier
features_list : list of dict
    List of features with coordinates and descriptions

Returns:
--------
pd.DataFrame : Comparative thermal analysis across features
"""
all_results = []
np.random.seed(42)  # For reproducibility

# ======================================================================
# Code Block 23
# ======================================================================

thermal_adjustments = {
    'waste_dump': lambda n: np.random.normal(3, 1, n),
    'tailings_dam': lambda n: np.random.uniform(12, 18, n),
    'facility': lambda n: np.zeros(n),
    'pit': lambda n: np.zeros(n)
}

for feature in features_list:

# ======================================================================
# Code Block 24
# ======================================================================

    pass
thermal = fetch_modis_lst_data(
        feature['lat'], 
        feature['lon'],
        '2022-01-01', 
        '2024-01-01'
    )
thermal['lst_day_celsius'] = kelvin_to_celsius(thermal['lst_day_kelvin'])
thermal['lst_night_celsius'] = kelvin_to_celsius(thermal['lst_night_kelvin'])

# ======================================================================
# Code Block 25
# ======================================================================

feature_type = feature['type']
recent = thermal['date'] > '2023-10-01'
adjustment_func = thermal_adjustments.get(feature_type, lambda n: np.zeros(n))
thermal.loc[recent, 'lst_day_celsius'] += adjustment_func(recent.sum())

# ======================================================================
# Code Block 26
# ======================================================================

baseline = calculate_thermal_baseline(thermal)
anomalies = detect_thermal_anomalies(thermal, baseline)

# ======================================================================
# Code Block 27
# ======================================================================

recent_period = anomalies['date'] > (anomalies['date'].max() - timedelta(days=90))
recent_anomalies = anomalies[recent_period]

# ======================================================================
# Code Block 28
# ======================================================================

max_score = recent_anomalies['anomaly_score'].max()
risk_level = pd.cut([max_score], bins=[-np.inf, 40, 60, np.inf], 
                       labels=['LOW', 'MEDIUM', 'HIGH'])[0]
    
feature_summary = {
        'site': site_name,
        'feature_name': feature['name'],
        'feature_type': feature['type'],
        'latitude': feature['lat'],
        'longitude': feature['lon'],
        'recent_mean_temp': recent_anomalies['lst_day_celsius'].mean(),
        'recent_max_temp': recent_anomalies['lst_day_celsius'].max(),
        'anomaly_count_90d': int(recent_anomalies['any_anomaly'].sum()),
        'max_anomaly_score': max_score,
        'mean_z_score': recent_anomalies['day_z_score'].mean(),
        'risk_level': risk_level
    }
    
all_results.append(feature_summary)

return pd.DataFrame(all_results)

# ======================================================================
# Code Block 29
# ======================================================================

print(f"{row['feature_name']} ({row['feature_type']}):")
print(f"  Risk Level: {row['risk_level']}")
print(f"  Recent Max Temp: {row['recent_max_temp']:.1f}°C")
print(f"  Anomalies (90d): {row['anomaly_count_90d']}")
print(f"  Max Anomaly Score: {row['max_anomaly_score']:.1f}/100")
print()

# ======================================================================
# Code Block 30
# ======================================================================

thermal_sorted = thermal_data.sort_values('date').copy()
thermal_sorted['rolling_mean'] = thermal_sorted['lst_day_celsius'].rolling(
    window=window_days // 8, min_periods=3  # 8-day composites
).mean()
thermal_sorted['rolling_max'] = thermal_sorted['lst_day_celsius'].rolling(
    window=window_days // 8, min_periods=3
).max()

# ======================================================================
# Code Block 31
# ======================================================================

recent_6mo = thermal_sorted[thermal_sorted['date'] > (thermal_sorted['date'].max() - timedelta(days=180))]

# ======================================================================
# Code Block 32
# ======================================================================

annual_trend = (
    np.polyfit(np.arange(len(recent_6mo)), recent_6mo['lst_day_celsius'].values, 1)[0] * (365 / 8)
    if len(recent_6mo) >= 10
    else 0
)

# ======================================================================
# Code Block 33
# ======================================================================

recent_30d = thermal_sorted[thermal_sorted['date'] > (thermal_sorted['date'].max() - timedelta(days=30))]
recent_mean = recent_30d['lst_day_celsius'].mean()
baseline_mean = baseline['overall']['day_mean']
deviation_from_baseline = recent_mean - baseline_mean

# ======================================================================
# Code Block 34
# ======================================================================

trend_bins = [-np.inf, -0.5, 0.5, 2, np.inf]
trend_labels = ['COOLING', 'STABLE', 'SLIGHT_WARMING', 'WARMING']

# ======================================================================
# Code Block 35
# ======================================================================

trend_category = pd.cut([annual_trend], bins=trend_bins, labels=trend_labels)[0]

# ======================================================================
# Code Block 36
# ======================================================================

urgency_map = {'COOLING': 'LOW', 'STABLE': 'LOW', 'SLIGHT_WARMING': 'MEDIUM', 'WARMING': 'HIGH'}
urgency_category = urgency_map[trend_category]

return {
    'annual_trend_celsius': annual_trend,
    'trend_status': trend_category,
    'urgency': urgency_category,
    'recent_mean': recent_mean,
    'baseline_mean': baseline_mean,
    'deviation': deviation_from_baseline,
    'recent_observations': len(recent_30d)
}

# ======================================================================
# Code Block 37
# ======================================================================

merged = pd.merge_asof(
    satellite_thermal[['date', 'lst_day_celsius']].sort_values('date'),
    ground_measurements[['date', 'ground_temp_celsius']].sort_values('date'),
    on='date',
    direction='nearest',
    tolerance=pd.Timedelta(days=4)
)

# ======================================================================
# Code Block 38
# ======================================================================

merged = merged.dropna()
if len(merged) < 10:
    return {'error': f'Insufficient matched observations: {len(merged)}'}

# ======================================================================
# Code Block 39
# ======================================================================

correlation = merged['lst_day_celsius'].corr(merged['ground_temp_celsius'])

# ======================================================================
# Code Block 40
# ======================================================================

merged['bias'] = merged['lst_day_celsius'] - merged['ground_temp_celsius']
mean_bias = merged['bias'].mean()
rmse = np.sqrt(np.mean(merged['bias'] ** 2))

# ======================================================================
# Code Block 41
# ======================================================================

ground_baseline = ground_measurements['ground_temp_celsius'].mean()
ground_std = ground_measurements['ground_temp_celsius'].std()
sat_threshold = satellite_thermal['lst_day_celsius'].mean() + 2.5 * satellite_thermal['lst_day_celsius'].std()

merged['ground_anomaly'] = (merged['ground_temp_celsius'] - ground_baseline) / ground_std > 2.5
merged['satellite_anomaly'] = merged['lst_day_celsius'] > sat_threshold

# ======================================================================
# Code Block 42
# ======================================================================

confusion = pd.crosstab(merged['satellite_anomaly'], merged['ground_anomaly'])

# ======================================================================
# Code Block 43
# ======================================================================

true_positives = confusion.get((True, True), 0)
false_positives = confusion.get((True, False), 0)
true_negatives = confusion.get((False, False), 0)
false_negatives = confusion.get((False, True), 0)

# ======================================================================
# Code Block 44
# ======================================================================

sensitivity = true_positives / max(1, true_positives + false_negatives)
specificity = true_negatives / max(1, true_negatives + false_positives)

return {
    'correlation': correlation,
    'mean_bias': mean_bias,
    'rmse': rmse,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'true_positives': int(true_positives),
    'false_positives': int(false_positives),
    'matched_observations': len(merged)
}

# ======================================================================
# Code Block 45
# ======================================================================

print(f"Error: {validation['error']}")

# ======================================================================
# Code Block 46
# ======================================================================

print(f"Correlation Coefficient: {validation['correlation']:.3f}")
print(f"Mean Bias: {validation['mean_bias']:+.2f}°C")
print(f"RMSE: {validation['rmse']:.2f}°C")
print(f"Anomaly Detection Sensitivity: {validation['sensitivity']:.1%}")
print(f"Anomaly Detection Specificity: {validation['specificity']:.1%}")
print(f"Matched Observations: {validation['matched_observations']}")
