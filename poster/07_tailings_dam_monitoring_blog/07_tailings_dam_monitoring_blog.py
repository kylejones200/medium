#!/usr/bin/env python3
"""
Python code extracted from 07_tailings_dam_monitoring_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

def fetch_sentinel2_data(latitude, longitude, start_date, end_date, cloud_threshold=20):
    """
    Fetch Sentinel-2 imagery for tailings dam monitoring.
    
    Uses Sentinel Hub or Google Earth Engine API to retrieve
    10m resolution multispectral data with cloud filtering.
    
    Parameters:
    -----------
    latitude : float
        Center latitude of tailings facility
    longitude : float
        Center longitude of tailings facility
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    cloud_threshold : float
        Maximum cloud cover percentage to accept
    
    Returns:
    --------
    pd.DataFrame : Time series of surface reflectance values
    """
    # This generates realistic synthetic Sentinel-2 data
    # In production, use Sentinel Hub API or Google Earth Engine
    
    dates = pd.date_range(start=start_date, end=end_date, freq='5D')  # Sentinel-2 revisit
    
    observations = []
    for date in dates:
        # Skip some observations for cloud cover
        if np.random.random() > (cloud_threshold / 100):
            # Sentinel-2 bands (surface reflectance 0-1)
            blue = 0.08 + np.random.normal(0, 0.01)
            green = 0.12 + np.random.normal(0, 0.01)
            red = 0.15 + np.random.normal(0, 0.01)
            nir = 0.35 + np.random.normal(0, 0.02)
            swir1 = 0.25 + np.random.normal(0, 0.015)
            swir2 = 0.18 + np.random.normal(0, 0.012)
            
            # Calculate indices
            ndvi = (nir - red) / (nir + red + 0.0001)
            ndwi = (green - nir) / (green + nir + 0.0001)
            ndti = (swir1 - swir2) / (swir1 + swir2 + 0.0001)  # Normalized Difference Turbidity Index
            
            observations.append({
                'date': date,
                'blue': blue,
                'green': green,
                'red': red,
                'nir': nir,
                'swir1': swir1,
                'swir2': swir2,
                'ndvi': ndvi,
                'ndwi': ndwi,
                'ndti': ndti,
                'cloud_cover': np.random.uniform(0, cloud_threshold),
                'latitude': latitude,
                'longitude': longitude
            })
    
    return pd.DataFrame(observations)

# Example: Fetch data for a tailings facility
facility_lat, facility_lon = -25.5, -50.3
imagery = fetch_sentinel2_data(facility_lat, facility_lon, '2022-01-01', '2024-01-01')

print(f"Collected {len(imagery)} cloud-free observations")
print(f"NDVI range: {imagery['ndvi'].min():.3f} to {imagery['ndvi'].max():.3f}")
print(f"NDWI range: {imagery['ndwi'].min():.3f} to {imagery['ndwi'].max():.3f}")
print(f"Mean cloud cover: {imagery['cloud_cover'].mean():.1f}%")

# ======================================================================
# Code Block 2
# ======================================================================

def detect_water_changes(imagery, baseline_start, baseline_end, analysis_start, analysis_end):
    """
    Detect water extent changes using NDWI analysis.
    
    Compares water-sensitive indices between baseline and analysis
    periods to identify expansion or contraction of ponded water.
    
    Parameters:
    -----------
    imagery : pd.DataFrame
        Sentinel-2 observations with NDWI calculated
    baseline_start, baseline_end : str
        Date range for establishing normal water extent
    analysis_start, analysis_end : str
        Date range for detecting changes
    
    Returns:
    --------
    dict : Water change analysis results
    """
    # Baseline period
    baseline = imagery[
        (imagery['date'] >= baseline_start) & 
        (imagery['date'] <= baseline_end)
    ]
    
    # Analysis period
    analysis = imagery[
        (imagery['date'] >= analysis_start) & 
        (imagery['date'] <= analysis_end)
    ]
    
    # Water detection threshold (NDWI > 0.3 typically indicates water)
    water_threshold = 0.3
    
    # Calculate water presence frequency in each period
    baseline_water_freq = (baseline['ndwi'] > water_threshold).mean()
    analysis_water_freq = (analysis['ndwi'] > water_threshold).mean()
    
    # Change metrics (Pythonic safe division)
    water_extent_change = analysis_water_freq - baseline_water_freq
    water_extent_change_pct = (water_extent_change / max(baseline_water_freq, 0.0001)) * 100
    
    # Turbidity analysis (NDTI)
    baseline_turbidity = baseline['ndti'].mean()
    analysis_turbidity = analysis['ndti'].mean()
    turbidity_change = analysis_turbidity - baseline_turbidity
    
    # NDWI trend in analysis period
    analysis['days_from_start'] = (analysis['date'] - analysis['date'].min()).dt.days
    if len(analysis) >= 10:
        trend_coefficients = np.polyfit(analysis['days_from_start'], analysis['ndwi'], 1)
        ndwi_trend_per_month = trend_coefficients[0] * 30
    else:
        ndwi_trend_per_month = 0
    
    # Risk assessment (Pythonic with scoring)
    water_risk_score = max(
        water_extent_change_pct,
        ndwi_trend_per_month * 400  # Scale trend to comparable range
    )
    risk_level = pd.cut([water_risk_score], 
                       bins=[-np.inf, 10, 20, np.inf],
                       labels=['LOW', 'MEDIUM', 'HIGH'])[0]
    
    return {
        'baseline_water_freq': baseline_water_freq,
        'analysis_water_freq': analysis_water_freq,
        'water_extent_change_pct': water_extent_change_pct,
        'baseline_turbidity': baseline_turbidity,
        'analysis_turbidity': analysis_turbidity,
        'turbidity_change': turbidity_change,
        'ndwi_trend_per_month': ndwi_trend_per_month,
        'risk_level': risk_level,
        'baseline_observations': len(baseline),
        'analysis_observations': len(analysis)
    }

# Analyze water changes
water_analysis = detect_water_changes(
    imagery,
    '2022-01-01', '2022-12-31',  # Baseline year
    '2023-10-01', '2024-01-01'   # Recent analysis period
)

print("Water Change Analysis:")
print("=" * 60)
print(f"Baseline Water Frequency: {water_analysis['baseline_water_freq']:.1%}")
print(f"Analysis Water Frequency: {water_analysis['analysis_water_freq']:.1%}")
print(f"Water Extent Change: {water_analysis['water_extent_change_pct']:+.1f}%")
print(f"NDWI Trend: {water_analysis['ndwi_trend_per_month']:+.4f} per month")
print(f"Turbidity Change: {water_analysis['turbidity_change']:+.3f}")
print(f"Risk Level: {water_analysis['risk_level']}")

# ======================================================================
# Code Block 3
# ======================================================================

def analyze_vegetation_changes(imagery, dam_area_pixels, buffer_area_pixels):
    """
    Analyze vegetation health changes around tailings facility.
    
    Compares NDVI patterns on dam structure versus surrounding
    areas to identify stress, die-off, or unusual growth patterns.
    
    Parameters:
    -----------
    imagery : pd.DataFrame
        Sentinel-2 observations with NDVI
    dam_area_pixels : list
        Pixels covering dam embankment
    buffer_area_pixels : list
        Control pixels in surrounding area
    
    Returns:
    --------
    dict : Vegetation analysis results
    """
    # Simulate spatial data (in production, use actual pixel coordinates)
    dam_ndvi_modifier = -0.05  # Dam shows vegetation stress
    
    imagery['dam_ndvi'] = imagery['ndvi'] + np.random.normal(dam_ndvi_modifier, 0.02, len(imagery))
    imagery['buffer_ndvi'] = imagery['ndvi'] + np.random.normal(0, 0.02, len(imagery))
    
    # Temporal analysis
    imagery['quarter'] = imagery['date'].dt.to_period('Q')
    quarterly = imagery.groupby('quarter').agg({
        'dam_ndvi': 'mean',
        'buffer_ndvi': 'mean',
        'ndvi': 'mean'
    }).reset_index()
    
    # Calculate relative vegetation stress (dam vs buffer)
    quarterly['relative_stress'] = (quarterly['buffer_ndvi'] - quarterly['dam_ndvi']) / quarterly['buffer_ndvi']
    
    # Recent stress level
    recent_quarters = quarterly.tail(4)
    current_stress = recent_quarters['relative_stress'].mean()
    stress_trend = np.polyfit(range(len(recent_quarters)), recent_quarters['relative_stress'], 1)[0]
    
    # Vegetation loss detection (NDVI decline)
    recent_data = imagery.tail(20)
    historical_data = imagery.head(50)
    
    recent_dam_ndvi = recent_data['dam_ndvi'].mean()
    historical_dam_ndvi = historical_data['dam_ndvi'].mean()
    ndvi_decline = historical_dam_ndvi - recent_dam_ndvi
    ndvi_decline_pct = (ndvi_decline / historical_dam_ndvi) * 100
    
    # Risk assessment (Pythonic with scoring)
    veg_risk_score = max(ndvi_decline_pct, current_stress * 100)
    veg_risk = pd.cut([veg_risk_score], 
                     bins=[-np.inf, 15, 25, np.inf],
                     labels=['LOW', 'MEDIUM', 'HIGH'])[0]
    
    return {
        'current_dam_ndvi': recent_dam_ndvi,
        'historical_dam_ndvi': historical_dam_ndvi,
        'ndvi_decline_pct': ndvi_decline_pct,
        'current_relative_stress': current_stress,
        'stress_trend': stress_trend,
        'vegetation_risk': veg_risk,
        'quarters_analyzed': len(quarterly)
    }

# Analyze vegetation
veg_analysis = analyze_vegetation_changes(imagery, dam_area_pixels=None, buffer_area_pixels=None)

print("\nVegetation Analysis:")
print("=" * 60)
print(f"Current Dam NDVI: {veg_analysis['current_dam_ndvi']:.3f}")
print(f"Historical Dam NDVI: {veg_analysis['historical_dam_ndvi']:.3f}")
print(f"NDVI Decline: {veg_analysis['ndvi_decline_pct']:.1f}%")
print(f"Relative Stress: {veg_analysis['current_relative_stress']:.1%}")
print(f"Stress Trend: {veg_analysis['stress_trend']:+.4f} per quarter")
print(f"Vegetation Risk: {veg_analysis['vegetation_risk']}")

# ======================================================================
# Code Block 4
# ======================================================================

def simulate_insar_displacement(num_dates=50, deformation_rate_mm_year=15):
    """
    Simulate InSAR displacement measurements for tailings dam.
    
    Real implementation would use Sentinel-1 SAR data processed
    through InSAR techniques to measure millimeter-scale ground motion.
    
    Parameters:
    -----------
    num_dates : int
        Number of acquisition dates
    deformation_rate_mm_year : float
        Annual deformation rate in millimeters
    
    Returns:
    --------
    pd.DataFrame : Displacement time series
    """
    dates = pd.date_range(start='2022-01-01', periods=num_dates, freq='12D')
    
    # Progressive deformation with noise
    days_elapsed = np.arange(num_dates) * 12
    years_elapsed = days_elapsed / 365
    
    # Linear deformation trend
    expected_displacement_mm = years_elapsed * deformation_rate_mm_year
    
    # Add measurement noise and atmospheric effects
    noise_mm = np.random.normal(0, 2, num_dates)  # 2mm standard error
    atmospheric_mm = np.random.normal(0, 3, num_dates) * np.sin(days_elapsed / 30)  # Seasonal atmospheric effects
    
    measured_displacement_mm = expected_displacement_mm + noise_mm + atmospheric_mm
    
    # Add acceleration in recent period (warning sign)
    recent_mask = num_dates - 10
    measured_displacement_mm[recent_mask:] += np.linspace(0, 20, 10)
    
    displacement_data = pd.DataFrame({
        'date': dates,
        'displacement_mm': measured_displacement_mm,
        'displacement_uncertainty_mm': np.abs(noise_mm) + 1.5,
        'coherence': np.random.uniform(0.7, 0.95, num_dates)  # InSAR coherence measure
    })
    
    return displacement_data

def analyze_deformation_trends(displacement_data, acceleration_window=90):
    """
    Analyze displacement trends to detect accelerating deformation.
    
    Parameters:
    -----------
    displacement_data : pd.DataFrame
        InSAR displacement time series
    acceleration_window : int
        Days to use for acceleration analysis
    
    Returns:
    --------
    dict : Deformation analysis results
    """
    # Overall displacement rate
    days_total = (displacement_data['date'].max() - displacement_data['date'].min()).days
    total_displacement = displacement_data['displacement_mm'].iloc[-1] - displacement_data['displacement_mm'].iloc[0]
    overall_rate_mm_year = (total_displacement / days_total) * 365
    
    # Recent period rate
    recent_cutoff = displacement_data['date'].max() - timedelta(days=acceleration_window)
    recent_data = displacement_data[displacement_data['date'] >= recent_cutoff]
    
    if len(recent_data) >= 5:
        days_recent = (recent_data['date'].max() - recent_data['date'].min()).days
        recent_displacement = recent_data['displacement_mm'].iloc[-1] - recent_data['displacement_mm'].iloc[0]
        recent_rate_mm_year = (recent_displacement / days_recent) * 365
        
        acceleration = recent_rate_mm_year - overall_rate_mm_year
    else:
        recent_rate_mm_year = overall_rate_mm_year
        acceleration = 0
    
    # Velocity change detection (Pythonic with pd.cut)
    # Create composite score considering both acceleration and overall rate
    deformation_score = max(acceleration, overall_rate_mm_year - 10)
    
    deformation_status_map = pd.cut([deformation_score], 
                                    bins=[-np.inf, 5, 10, 15, np.inf],
                                    labels=['NORMAL', 'STEADY_HIGH', 'INCREASING', 'ACCELERATING'])[0]
    
    urgency_map = {
        'NORMAL': 'LOW',
        'STEADY_HIGH': 'MEDIUM', 
        'INCREASING': 'HIGH',
        'ACCELERATING': 'CRITICAL'
    }
    deformation_status = deformation_status_map
    urgency = urgency_map[deformation_status]
    
    # Confidence assessment (Pythonic with pd.cut)
    mean_coherence = displacement_data['coherence'].mean()
    confidence = pd.cut([mean_coherence], 
                       bins=[0, 0.70, 0.85, 1.0],
                       labels=['LOW', 'MEDIUM', 'HIGH'])[0]
    
    return {
        'total_displacement_mm': total_displacement,
        'overall_rate_mm_year': overall_rate_mm_year,
        'recent_rate_mm_year': recent_rate_mm_year,
        'acceleration_mm_year2': acceleration,
        'deformation_status': deformation_status,
        'urgency': urgency,
        'mean_coherence': mean_coherence,
        'confidence': confidence,
        'measurements': len(displacement_data)
    }

# Simulate and analyze deformation
displacement = simulate_insar_displacement(num_dates=60, deformation_rate_mm_year=15)
deformation = analyze_deformation_trends(displacement)

print("\nDeformation Analysis:")
print("=" * 60)
print(f"Total Displacement: {deformation['total_displacement_mm']:.1f} mm")
print(f"Overall Rate: {deformation['overall_rate_mm_year']:.1f} mm/year")
print(f"Recent Rate: {deformation['recent_rate_mm_year']:.1f} mm/year")
print(f"Acceleration: {deformation['acceleration_mm_year2']:+.1f} mm/year²")
print(f"Status: {deformation['deformation_status']}")
print(f"Urgency: {deformation['urgency']}")
print(f"Measurement Confidence: {deformation['confidence']}")

# ======================================================================
# Code Block 5
# ======================================================================

def integrated_dam_assessment(water_analysis, veg_analysis, deformation, imagery_data):
    """
    Integrate multiple monitoring parameters for comprehensive risk assessment.
    
    Combines water extent, vegetation health, and deformation data
    to produce unified risk scoring and prioritized recommendations.
    
    Parameters:
    -----------
    water_analysis : dict
        Water change detection results
    veg_analysis : dict
        Vegetation analysis results
    deformation : dict
        Deformation analysis results
    imagery_data : pd.DataFrame
        Raw imagery data for temporal analysis
    
    Returns:
    --------
    dict : Integrated assessment with overall risk score
    """
    # Risk scoring system (0-100 scale)
    risk_weights = {
        'water': 0.30,
        'vegetation': 0.25,
        'deformation': 0.45  # Highest weight - direct structural indicator
    }
    
    # Water risk score (Pythonic with dictionary mapping)
    risk_level_scores = {'LOW': 20, 'MEDIUM': 50, 'HIGH': 80}
    water_risk_score = risk_level_scores[water_analysis['risk_level']]
    
    # Add points for specific conditions (Pythonic chaining)
    water_risk_score += (
        (15 if water_analysis['water_extent_change_pct'] > 30 else 0) +
        (15 if water_analysis['ndwi_trend_per_month'] > 0.08 else 0)
    )
    water_risk_score = min(100, water_risk_score)
    
    # Vegetation risk score (Pythonic with dictionary mapping)
    veg_risk_scores = {'LOW': 15, 'MEDIUM': 45, 'HIGH': 75}
    veg_risk_score = veg_risk_scores[veg_analysis['vegetation_risk']]
    
    # Add bonus for severe decline
    veg_risk_score += (20 if veg_analysis['ndvi_decline_pct'] > 30 else 0)
    veg_risk_score = min(100, veg_risk_score)
    
    # Deformation risk score (Pythonic with dictionary mapping)
    deform_urgency_scores = {'LOW': 15, 'MEDIUM': 45, 'HIGH': 75, 'CRITICAL': 95}
    deform_risk_score = deform_urgency_scores[deformation['urgency']]
    
    # Adjust for severe acceleration
    deform_risk_score += (20 if deformation['acceleration_mm_year2'] > 20 else 0)
    deform_risk_score = min(100, deform_risk_score)
    
    # Calculate weighted overall risk
    overall_risk_score = (
        water_risk_score * risk_weights['water'] +
        veg_risk_score * risk_weights['vegetation'] +
        deform_risk_score * risk_weights['deformation']
    )
    
    # Overall risk classification (Pythonic with pd.cut and dictionaries)
    overall_risk_level = pd.cut([overall_risk_score],
                                bins=[-np.inf, 40, 60, 75, np.inf],
                                labels=['NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL'])[0]
    
    # Action mappings (Pythonic)
    action_map = {
        'NORMAL': 'Continue routine monitoring',
        'ELEVATED': 'Schedule inspection within 30 days - Continue monitoring',
        'HIGH': 'Schedule inspection within 1 week - Increase monitoring frequency',
        'CRITICAL': 'IMMEDIATE INSPECTION REQUIRED - Consider emergency drawdown'
    }
    priority_map = {'NORMAL': 4, 'ELEVATED': 3, 'HIGH': 2, 'CRITICAL': 1}
    
    recommended_action = action_map[overall_risk_level]
    inspection_priority = priority_map[overall_risk_level]
    
    # Identify primary concern
    component_scores = {
        'deformation': deform_risk_score,
        'water': water_risk_score,
        'vegetation': veg_risk_score
    }
    primary_concern = max(component_scores, key=component_scores.get)
    
    return {
        'overall_risk_score': overall_risk_score,
        'overall_risk_level': overall_risk_level,
        'water_risk_score': water_risk_score,
        'vegetation_risk_score': veg_risk_score,
        'deformation_risk_score': deform_risk_score,
        'primary_concern': primary_concern,
        'recommended_action': recommended_action,
        'inspection_priority': inspection_priority,
        'assessment_date': datetime.now(),
        'data_quality': deformation['confidence']
    }

# Perform integrated assessment
assessment = integrated_dam_assessment(water_analysis, veg_analysis, deformation, imagery)

print("\nIntegrated Dam Safety Assessment:")
print("=" * 70)
print(f"Overall Risk Score: {assessment['overall_risk_score']:.1f}/100")
print(f"Risk Level: {assessment['overall_risk_level']}")
print(f"\nComponent Scores:")
print(f"  Deformation: {assessment['deformation_risk_score']:.1f}/100")
print(f"  Water Extent: {assessment['water_risk_score']:.1f}/100")
print(f"  Vegetation: {assessment['vegetation_risk_score']:.1f}/100")
print(f"\nPrimary Concern: {assessment['primary_concern'].upper()}")
print(f"Inspection Priority: P{assessment['inspection_priority']}")
print(f"\nRecommended Action:")
print(f"  {assessment['recommended_action']}")
print(f"\nData Quality: {assessment['data_quality']}")

# ======================================================================
# Code Block 6
# ======================================================================

def generate_monitoring_report(facility_name, assessment, water_analysis, veg_analysis, deformation):
    """
    Generate standardized monitoring report for management review.
    
    Creates summary suitable for regulatory submissions,
    board presentations, and operational decisions.
    """
    report = {
        'facility_name': facility_name,
        'report_date': datetime.now().strftime('%Y-%m-%d'),
        'reporting_period': '90 days',
        'overall_assessment': {
            'risk_level': assessment['overall_risk_level'],
            'risk_score': f"{assessment['overall_risk_score']:.1f}",
            'primary_concern': assessment['primary_concern'],
            'action_required': assessment['recommended_action']
        },
        'deformation_summary': {
            'total_displacement_mm': f"{deformation['total_displacement_mm']:.1f}",
            'rate_mm_year': f"{deformation['recent_rate_mm_year']:.1f}",
            'status': deformation['deformation_status'],
            'urgency': deformation['urgency']
        },
        'water_summary': {
            'extent_change_pct': f"{water_analysis['water_extent_change_pct']:+.1f}",
            'trend_per_month': f"{water_analysis['ndwi_trend_per_month']:+.4f}",
            'risk': water_analysis['risk_level']
        },
        'vegetation_summary': {
            'ndvi_decline_pct': f"{veg_analysis['ndvi_decline_pct']:.1f}",
            'relative_stress': f"{veg_analysis['current_relative_stress']:.1%}",
            'risk': veg_analysis['vegetation_risk']
        },
        'data_quality': assessment['data_quality'],
        'next_assessment': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    }
    
    return report

# Generate report
report = generate_monitoring_report('Tailings Storage Facility #3', assessment, water_analysis, veg_analysis, deformation)

print("\n" + "=" * 70)
print("TAILINGS DAM MONITORING REPORT")
print("=" * 70)
print(f"Facility: {report['facility_name']}")
print(f"Report Date: {report['report_date']}")
print(f"Reporting Period: {report['reporting_period']}")
print(f"\nOVERALL ASSESSMENT:")
print(f"  Risk Level: {report['overall_assessment']['risk_level']}")
print(f"  Risk Score: {report['overall_assessment']['risk_score']}/100")
print(f"  Primary Concern: {report['overall_assessment']['primary_concern']}")
print(f"  Action Required: {report['overall_assessment']['action_required']}")
print(f"\nDEFORMATION:")
print(f"  Displacement: {report['deformation_summary']['total_displacement_mm']} mm")
print(f"  Rate: {report['deformation_summary']['rate_mm_year']} mm/year")
print(f"  Status: {report['deformation_summary']['status']}")
print(f"\nWATER EXTENT:")
print(f"  Change: {report['water_summary']['extent_change_pct']}%")
print(f"  Trend: {report['water_summary']['trend_per_month']} NDWI/month")
print(f"\nVEGETATION:")
print(f"  NDVI Decline: {report['vegetation_summary']['ndvi_decline_pct']}%")
print(f"  Relative Stress: {report['vegetation_summary']['relative_stress']}")
print(f"\nNext Assessment: {report['next_assessment']}")

# ======================================================================
# Code Block 7
# ======================================================================

dates = pd.date_range(start=start_date, end=end_date, freq='5D')  # Sentinel-2 revisit

observations = []
for date in dates:

# ======================================================================
# Code Block 8
# ======================================================================

blue = 0.08 + np.random.normal(0, 0.01)
        green = 0.12 + np.random.normal(0, 0.01)
        red = 0.15 + np.random.normal(0, 0.01)
        nir = 0.35 + np.random.normal(0, 0.02)
        swir1 = 0.25 + np.random.normal(0, 0.015)
        swir2 = 0.18 + np.random.normal(0, 0.012)

# ======================================================================
# Code Block 9
# ======================================================================

ndvi = (nir - red) / (nir + red + 0.0001)
        ndwi = (green - nir) / (green + nir + 0.0001)
        ndti = (swir1 - swir2) / (swir1 + swir2 + 0.0001)  # Normalized Difference Turbidity Index
        
        observations.append({
            'date': date,
            'blue': blue,
            'green': green,
            'red': red,
            'nir': nir,
            'swir1': swir1,
            'swir2': swir2,
            'ndvi': ndvi,
            'ndwi': ndwi,
            'ndti': ndti,
            'cloud_cover': np.random.uniform(0, cloud_threshold),
            'latitude': latitude,
            'longitude': longitude
        })

return pd.DataFrame(observations)

# ======================================================================
# Code Block 10
# ======================================================================

baseline = imagery[
    (imagery['date'] >= baseline_start) & 
    (imagery['date'] <= baseline_end)
]

# ======================================================================
# Code Block 11
# ======================================================================

analysis = imagery[
    (imagery['date'] >= analysis_start) & 
    (imagery['date'] <= analysis_end)
]

# ======================================================================
# Code Block 12
# ======================================================================

water_threshold = 0.3

# ======================================================================
# Code Block 13
# ======================================================================

baseline_water_freq = (baseline['ndwi'] > water_threshold).mean()
analysis_water_freq = (analysis['ndwi'] > water_threshold).mean()

# ======================================================================
# Code Block 14
# ======================================================================

water_extent_change = analysis_water_freq - baseline_water_freq
water_extent_change_pct = (water_extent_change / max(baseline_water_freq, 0.0001)) * 100

# ======================================================================
# Code Block 15
# ======================================================================

baseline_turbidity = baseline['ndti'].mean()
analysis_turbidity = analysis['ndti'].mean()
turbidity_change = analysis_turbidity - baseline_turbidity

# ======================================================================
# Code Block 16
# ======================================================================

analysis['days_from_start'] = (analysis['date'] - analysis['date'].min()).dt.days
if len(analysis) >= 10:
    trend_coefficients = np.polyfit(analysis['days_from_start'], analysis['ndwi'], 1)
    ndwi_trend_per_month = trend_coefficients[0] * 30
else:
    ndwi_trend_per_month = 0

# ======================================================================
# Code Block 17
# ======================================================================

water_risk_score = max(
    water_extent_change_pct,
    ndwi_trend_per_month * 400  # Scale trend to comparable range
)
risk_level = pd.cut([water_risk_score], 
                   bins=[-np.inf, 10, 20, np.inf],
                   labels=['LOW', 'MEDIUM', 'HIGH'])[0]

return {
    'baseline_water_freq': baseline_water_freq,
    'analysis_water_freq': analysis_water_freq,
    'water_extent_change_pct': water_extent_change_pct,
    'baseline_turbidity': baseline_turbidity,
    'analysis_turbidity': analysis_turbidity,
    'turbidity_change': turbidity_change,
    'ndwi_trend_per_month': ndwi_trend_per_month,
    'risk_level': risk_level,
    'baseline_observations': len(baseline),
    'analysis_observations': len(analysis)
}

# ======================================================================
# Code Block 18
# ======================================================================

dam_ndvi_modifier = -0.05  # Dam shows vegetation stress

imagery['dam_ndvi'] = imagery['ndvi'] + np.random.normal(dam_ndvi_modifier, 0.02, len(imagery))
imagery['buffer_ndvi'] = imagery['ndvi'] + np.random.normal(0, 0.02, len(imagery))

# ======================================================================
# Code Block 19
# ======================================================================

imagery['quarter'] = imagery['date'].dt.to_period('Q')
quarterly = imagery.groupby('quarter').agg({
    'dam_ndvi': 'mean',
    'buffer_ndvi': 'mean',
    'ndvi': 'mean'
}).reset_index()

# ======================================================================
# Code Block 20
# ======================================================================

quarterly['relative_stress'] = (quarterly['buffer_ndvi'] - quarterly['dam_ndvi']) / quarterly['buffer_ndvi']

# ======================================================================
# Code Block 21
# ======================================================================

recent_quarters = quarterly.tail(4)
current_stress = recent_quarters['relative_stress'].mean()
stress_trend = np.polyfit(range(len(recent_quarters)), recent_quarters['relative_stress'], 1)[0]

# ======================================================================
# Code Block 22
# ======================================================================

recent_data = imagery.tail(20)
historical_data = imagery.head(50)

recent_dam_ndvi = recent_data['dam_ndvi'].mean()
historical_dam_ndvi = historical_data['dam_ndvi'].mean()
ndvi_decline = historical_dam_ndvi - recent_dam_ndvi
ndvi_decline_pct = (ndvi_decline / historical_dam_ndvi) * 100

# ======================================================================
# Code Block 23
# ======================================================================

veg_risk_score = max(ndvi_decline_pct, current_stress * 100)
veg_risk = pd.cut([veg_risk_score], 
                 bins=[-np.inf, 15, 25, np.inf],
                 labels=['LOW', 'MEDIUM', 'HIGH'])[0]

return {
    'current_dam_ndvi': recent_dam_ndvi,
    'historical_dam_ndvi': historical_dam_ndvi,
    'ndvi_decline_pct': ndvi_decline_pct,
    'current_relative_stress': current_stress,
    'stress_trend': stress_trend,
    'vegetation_risk': veg_risk,
    'quarters_analyzed': len(quarterly)
}

# ======================================================================
# Code Block 24
# ======================================================================

"""
Simulate InSAR displacement measurements for tailings dam.

Real implementation would use Sentinel-1 SAR data processed
through InSAR techniques to measure millimeter-scale ground motion.

Parameters:
-----------
num_dates : int
    Number of acquisition dates
deformation_rate_mm_year : float
    Annual deformation rate in millimeters

Returns:
--------
pd.DataFrame : Displacement time series
"""
dates = pd.date_range(start='2022-01-01', periods=num_dates, freq='12D')

# ======================================================================
# Code Block 25
# ======================================================================

days_elapsed = np.arange(num_dates) * 12
years_elapsed = days_elapsed / 365

# ======================================================================
# Code Block 26
# ======================================================================

expected_displacement_mm = years_elapsed * deformation_rate_mm_year

# ======================================================================
# Code Block 27
# ======================================================================

noise_mm = np.random.normal(0, 2, num_dates)  # 2mm standard error
atmospheric_mm = np.random.normal(0, 3, num_dates) * np.sin(days_elapsed / 30)  # Seasonal atmospheric effects

measured_displacement_mm = expected_displacement_mm + noise_mm + atmospheric_mm

# ======================================================================
# Code Block 28
# ======================================================================

recent_mask = num_dates - 10
measured_displacement_mm[recent_mask:] += np.linspace(0, 20, 10)

displacement_data = pd.DataFrame({
    'date': dates,
    'displacement_mm': measured_displacement_mm,
    'displacement_uncertainty_mm': np.abs(noise_mm) + 1.5,
    'coherence': np.random.uniform(0.7, 0.95, num_dates)  # InSAR coherence measure
})

return displacement_data

# ======================================================================
# Code Block 29
# ======================================================================

days_total = (displacement_data['date'].max() - displacement_data['date'].min()).days
total_displacement = displacement_data['displacement_mm'].iloc[-1] - displacement_data['displacement_mm'].iloc[0]
overall_rate_mm_year = (total_displacement / days_total) * 365

# ======================================================================
# Code Block 30
# ======================================================================

recent_cutoff = displacement_data['date'].max() - timedelta(days=acceleration_window)
recent_data = displacement_data[displacement_data['date'] >= recent_cutoff]

if len(recent_data) >= 5:
    days_recent = (recent_data['date'].max() - recent_data['date'].min()).days
    recent_displacement = recent_data['displacement_mm'].iloc[-1] - recent_data['displacement_mm'].iloc[0]
    recent_rate_mm_year = (recent_displacement / days_recent) * 365
    
    acceleration = recent_rate_mm_year - overall_rate_mm_year
else:
    recent_rate_mm_year = overall_rate_mm_year
    acceleration = 0

# ======================================================================
# Code Block 31
# ======================================================================

deformation_score = max(acceleration, overall_rate_mm_year - 10)

deformation_status_map = pd.cut([deformation_score], 
                                bins=[-np.inf, 5, 10, 15, np.inf],
                                labels=['NORMAL', 'STEADY_HIGH', 'INCREASING', 'ACCELERATING'])[0]

urgency_map = {
    'NORMAL': 'LOW',
    'STEADY_HIGH': 'MEDIUM', 
    'INCREASING': 'HIGH',
    'ACCELERATING': 'CRITICAL'
}
deformation_status = deformation_status_map
urgency = urgency_map[deformation_status]

# ======================================================================
# Code Block 32
# ======================================================================

mean_coherence = displacement_data['coherence'].mean()
confidence = pd.cut([mean_coherence], 
                   bins=[0, 0.70, 0.85, 1.0],
                   labels=['LOW', 'MEDIUM', 'HIGH'])[0]

return {
    'total_displacement_mm': total_displacement,
    'overall_rate_mm_year': overall_rate_mm_year,
    'recent_rate_mm_year': recent_rate_mm_year,
    'acceleration_mm_year2': acceleration,
    'deformation_status': deformation_status,
    'urgency': urgency,
    'mean_coherence': mean_coherence,
    'confidence': confidence,
    'measurements': len(displacement_data)
}

# ======================================================================
# Code Block 33
# ======================================================================

risk_weights = {
    'water': 0.30,
    'vegetation': 0.25,
    'deformation': 0.45  # Highest weight - direct structural indicator
}

# ======================================================================
# Code Block 34
# ======================================================================

risk_level_scores = {'LOW': 20, 'MEDIUM': 50, 'HIGH': 80}
water_risk_score = risk_level_scores[water_analysis['risk_level']]

# ======================================================================
# Code Block 35
# ======================================================================

water_risk_score += (
    (15 if water_analysis['water_extent_change_pct'] > 30 else 0) +
    (15 if water_analysis['ndwi_trend_per_month'] > 0.08 else 0)
)
water_risk_score = min(100, water_risk_score)

# ======================================================================
# Code Block 36
# ======================================================================

veg_risk_scores = {'LOW': 15, 'MEDIUM': 45, 'HIGH': 75}
veg_risk_score = veg_risk_scores[veg_analysis['vegetation_risk']]

# ======================================================================
# Code Block 37
# ======================================================================

veg_risk_score += (20 if veg_analysis['ndvi_decline_pct'] > 30 else 0)
veg_risk_score = min(100, veg_risk_score)

# ======================================================================
# Code Block 38
# ======================================================================

deform_urgency_scores = {'LOW': 15, 'MEDIUM': 45, 'HIGH': 75, 'CRITICAL': 95}
deform_risk_score = deform_urgency_scores[deformation['urgency']]

# ======================================================================
# Code Block 39
# ======================================================================

deform_risk_score += (20 if deformation['acceleration_mm_year2'] > 20 else 0)
deform_risk_score = min(100, deform_risk_score)

# ======================================================================
# Code Block 40
# ======================================================================

overall_risk_score = (
    water_risk_score * risk_weights['water'] +
    veg_risk_score * risk_weights['vegetation'] +
    deform_risk_score * risk_weights['deformation']
)

# ======================================================================
# Code Block 41
# ======================================================================

overall_risk_level = pd.cut([overall_risk_score],
                            bins=[-np.inf, 40, 60, 75, np.inf],
                            labels=['NORMAL', 'ELEVATED', 'HIGH', 'CRITICAL'])[0]

# ======================================================================
# Code Block 42
# ======================================================================

action_map = {
    'NORMAL': 'Continue routine monitoring',
    'ELEVATED': 'Schedule inspection within 30 days - Continue monitoring',
    'HIGH': 'Schedule inspection within 1 week - Increase monitoring frequency',
    'CRITICAL': 'IMMEDIATE INSPECTION REQUIRED - Consider emergency drawdown'
}
priority_map = {'NORMAL': 4, 'ELEVATED': 3, 'HIGH': 2, 'CRITICAL': 1}

recommended_action = action_map[overall_risk_level]
inspection_priority = priority_map[overall_risk_level]

# ======================================================================
# Code Block 43
# ======================================================================

component_scores = {
    'deformation': deform_risk_score,
    'water': water_risk_score,
    'vegetation': veg_risk_score
}
primary_concern = max(component_scores, key=component_scores.get)

return {
    'overall_risk_score': overall_risk_score,
    'overall_risk_level': overall_risk_level,
    'water_risk_score': water_risk_score,
    'vegetation_risk_score': veg_risk_score,
    'deformation_risk_score': deform_risk_score,
    'primary_concern': primary_concern,
    'recommended_action': recommended_action,
    'inspection_priority': inspection_priority,
    'assessment_date': datetime.now(),
    'data_quality': deformation['confidence']
}

# ======================================================================
# Code Block 44
# ======================================================================

"""
Generate standardized monitoring report for management review.

Creates summary suitable for regulatory submissions,
board presentations, and operational decisions.
"""
report = {
    'facility_name': facility_name,
    'report_date': datetime.now().strftime('%Y-%m-%d'),
    'reporting_period': '90 days',
    'overall_assessment': {
        'risk_level': assessment['overall_risk_level'],
        'risk_score': f"{assessment['overall_risk_score']:.1f}",
        'primary_concern': assessment['primary_concern'],
        'action_required': assessment['recommended_action']
    },
    'deformation_summary': {
        'total_displacement_mm': f"{deformation['total_displacement_mm']:.1f}",
        'rate_mm_year': f"{deformation['recent_rate_mm_year']:.1f}",
        'status': deformation['deformation_status'],
        'urgency': deformation['urgency']
    },
    'water_summary': {
        'extent_change_pct': f"{water_analysis['water_extent_change_pct']:+.1f}",
        'trend_per_month': f"{water_analysis['ndwi_trend_per_month']:+.4f}",
        'risk': water_analysis['risk_level']
    },
    'vegetation_summary': {
        'ndvi_decline_pct': f"{veg_analysis['ndvi_decline_pct']:.1f}",
        'relative_stress': f"{veg_analysis['current_relative_stress']:.1%}",
        'risk': veg_analysis['vegetation_risk']
    },
    'data_quality': assessment['data_quality'],
    'next_assessment': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
}

return report
