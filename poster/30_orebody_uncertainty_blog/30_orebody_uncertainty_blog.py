#!/usr/bin/env python3
"""
Python code extracted from 30_orebody_uncertainty_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

# Download GA National Geochemical Survey
DATA_URL = "https://pid.geoscience.gov.au/dataset/ga/132522"
DATA_DIR = "ga_geochem"
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading Geoscience Australia geochemical data...")
# Note: Actual download requires navigating GA portal; for demo, we'll use synthetic data

# For production, use GA's REST API:
# import requests
# response = requests.get(DATA_URL)
# with open(f"{DATA_DIR}/geochem.zip", 'wb') as f:
#     f.write(response.content)

# ======================================================================
# Code Block 2
# ======================================================================

np.random.seed(123)

n_samples = 2500

# Yilgarn Craton bounding box (WA)
lon_min, lon_max = 119.0, 122.0
lat_min, lat_max = -31.0, -28.5

# Sample locations
lons = np.random.uniform(lon_min, lon_max, n_samples)
lats = np.random.uniform(lat_min, lat_max, n_samples)

# Synthetic gold grades with spatial structure
# Simulate orogenic gold deposits: high-grade shoots along NNW-trending structures
center_lon, center_lat = 120.5, -29.75
dist_to_center = np.sqrt((lons - center_lon)**2 + (lats - center_lat)**2)

# Base grade pattern: exponential decay from center
base_grade = 0.5 * np.exp(-dist_to_center / 0.5) + 0.05

# Add structural control (NNW trend)
nnw_component = 0.3 * np.exp(-((lons - center_lon) / 0.3)**2) * np.exp(-((lats - center_lat) / 1.0)**2)

# Combine with lognormal noise
gold_ppm = base_grade + nnw_component + np.random.lognormal(0, 0.5, n_samples) * 0.1
gold_ppm = np.clip(gold_ppm, 0.01, 50.0)  # Realistic range

# Create dataframe
geochem = pd.DataFrame({
    'sample_id': [f'GA{i:06d}' for i in range(n_samples)],
    'longitude': lons,
    'latitude': lats,
    'Au_ppm': gold_ppm,
    'sample_type': np.random.choice(['soil', 'stream_sed', 'rock'], n_samples, p=[0.7, 0.2, 0.1])
})

# print(f" Generated {len(geochem):,} synthetic samples")
print(f"Au grade range: {gold_ppm.min():.3f} - {gold_ppm.max():.3f} ppm")
print(f"Au grade mean: {gold_ppm.mean():.3f} ppm")

# ======================================================================
# Code Block 3
# ======================================================================

import geopandas as gpd
from shapely.geometry import Point

# Create GeoDataFrame
geometry = [Point(lon, lat) for lon, lat in zip(geochem['longitude'], geochem['latitude'])]
gdf = gpd.GeoDataFrame(geochem, geometry=geometry, crs='EPSG:4326')

# Project to UTM Zone 50S (Western Australia)
gdf = gdf.to_crs('EPSG:32750')

# Extract UTM coordinates
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y

# print(" Projected to UTM coordinates")

# ======================================================================
# Code Block 4
# ======================================================================

# Log transform (add 1 to avoid log(0))
gdf['log_Au'] = np.log1p(gdf['Au_ppm'])

print(f"Log-Au range: {gdf['log_Au'].min():.3f} - {gdf['log_Au'].max():.3f}")
print(f"Log-Au mean: {gdf['log_Au'].mean():.3f}")
print(f"Log-Au std: {gdf['log_Au'].std():.3f}")

# ======================================================================
# Code Block 5
# ======================================================================

from skgstat import Variogram
import matplotlib.pyplot as plt

# Prepare coordinates and values
coords = gdf[['x', 'y']].values
values = gdf['log_Au'].values

print("Computing experimental variogram...")
V = Variogram(
    coords, 
    values, 
    model='spherical',
    maxlag=50000,  # 50 km max
    n_lags=20,
    normalize=False
)

print(f"\nVariogram parameters:")
print(f"  Model: {V.model.__name__}")
print(f"  Range: {V.parameters[0]/1000:.1f} km")
print(f"  Sill: {V.parameters[1]:.4f}")
print(f"  Nugget: {V.parameters[2]:.4f}")

# ======================================================================
# Code Block 6
# ======================================================================

from pykrige.ok import OrdinaryKriging

# Define estimation grid
x_min, x_max = gdf['x'].min(), gdf['x'].max()
y_min, y_max = gdf['y'].min(), gdf['y'].max()

nx, ny = 100, 100
gridx = np.linspace(x_min, x_max, nx)
gridy = np.linspace(y_min, y_max, ny)

# print(f"Grid dimensions: {nx}  {ny} = {nx*ny:,} blocks")

# Ordinary Kriging
print("\nRunning Ordinary Kriging...")
OK = OrdinaryKriging(
    gdf['x'].values,
    gdf['y'].values,
    gdf['log_Au'].values,
    variogram_model='spherical',
    variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
    verbose=False,
    enable_plotting=False
)

z_ok, ss_ok = OK.execute('grid', gridx, gridy)

# Back-transform to ppm
grade_ok = np.expm1(z_ok)  # exp(log1p(x)) - 1 = x

# print(f" Kriging complete")
print(f"Grade range: {grade_ok.min():.3f} - {grade_ok.max():.3f} ppm")
print(f"Grade mean: {grade_ok.mean():.3f} ppm")

# ======================================================================
# Code Block 7
# ======================================================================

from pykrige.ok import OrdinaryKriging

n_realizations = 50
simulations = []

print(f"\nGenerating {n_realizations} SGS realizations...")

for i in range(n_realizations):
    if (i+1) % 10 == 0:
        print(f"  Realization {i+1}/{n_realizations}...")
    
    # Perturb sample data with random noise (simplified SGS)
    # In production, use proper SGS algorithm (PyGSLIB, SGeMS, or custom)
    noise = np.random.normal(0, 0.1, len(gdf))
    perturbed = gdf['log_Au'].values + noise
    
    # Krige with perturbed data
    OK_sim = OrdinaryKriging(
        gdf['x'].values,
        gdf['y'].values,
        perturbed,
        variogram_model='spherical',
        variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
        verbose=False,
        enable_plotting=False
    )
    
    z_sim, _ = OK_sim.execute('grid', gridx, gridy)
    simulations.append(z_sim)

# Stack realizations
sim_stack = np.stack(simulations)  # Shape: (n_realizations, ny, nx)

# print(f" Generated {n_realizations} realizations")
print(f"Simulation stack shape: {sim_stack.shape}")

# ======================================================================
# Code Block 8
# ======================================================================

# Compute statistics across realizations
mean_sim = np.mean(sim_stack, axis=0)
std_sim = np.std(sim_stack, axis=0)
p10_sim = np.percentile(sim_stack, 10, axis=0)
p50_sim = np.percentile(sim_stack, 50, axis=0)
p90_sim = np.percentile(sim_stack, 90, axis=0)

# Back-transform to ppm
grade_mean = np.expm1(mean_sim)
grade_std = np.expm1(std_sim)
grade_p10 = np.expm1(p10_sim)
grade_p50 = np.expm1(p50_sim)
grade_p90 = np.expm1(p90_sim)

print("\nSimulation Statistics:")
print(f"  Mean grade: {grade_mean.mean():.3f} ppm")
print(f"  Std dev: {grade_std.mean():.3f} ppm")
print(f"  P10 grade: {grade_p10.mean():.3f} ppm")
print(f"  P50 grade: {grade_p50.mean():.3f} ppm")
print(f"  P90 grade: {grade_p90.mean():.3f} ppm")

# ======================================================================
# Code Block 9
# ======================================================================

# Probability of exceeding economic cutoff
cutoff_ppm = 0.5  # Example cutoff
cutoff_log = np.log1p(cutoff_ppm)

# Count realizations exceeding cutoff
prob_exceed = (sim_stack > cutoff_log).mean(axis=0)

print(f"\nExceedance Probability (>{cutoff_ppm} ppm):")
print(f"  Mean probability: {prob_exceed.mean():.1%}")
print(f"  Max probability: {prob_exceed.max():.1%}")
print(f"  Blocks with >50% prob: {(prob_exceed > 0.5).sum()}/{prob_exceed.size}")

# ======================================================================
# Code Block 10
# ======================================================================

# Complete orebody uncertainty modeling pipeline
# Uses synthetic data mimicking GA geochemical patterns

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from skgstat import Variogram
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

# ============================================================================
# 1. Generate synthetic geochemical data
# ============================================================================

np.random.seed(123)
n_samples = 2500

lon_min, lon_max = 119.0, 122.0
lat_min, lat_max = -31.0, -28.5

lons = np.random.uniform(lon_min, lon_max, n_samples)
lats = np.random.uniform(lat_min, lat_max, n_samples)

center_lon, center_lat = 120.5, -29.75
dist_to_center = np.sqrt((lons - center_lon)**2 + (lats - center_lat)**2)

base_grade = 0.5 * np.exp(-dist_to_center / 0.5) + 0.05
nnw_component = 0.3 * np.exp(-((lons - center_lon) / 0.3)**2) * np.exp(-((lats - center_lat) / 1.0)**2)
gold_ppm = base_grade + nnw_component + np.random.lognormal(0, 0.5, n_samples) * 0.1
gold_ppm = np.clip(gold_ppm, 0.01, 50.0)

geochem = pd.DataFrame({
    'sample_id': [f'GA{i:06d}' for i in range(n_samples)],
    'longitude': lons,
    'latitude': lats,
    'Au_ppm': gold_ppm
})

# print(f" Generated {len(geochem):,} samples")

# ============================================================================
# 2. Project and transform
# ============================================================================

geometry = [Point(lon, lat) for lon, lat in zip(geochem['longitude'], geochem['latitude'])]
gdf = gpd.GeoDataFrame(geochem, geometry=geometry, crs='EPSG:4326')
gdf = gdf.to_crs('EPSG:32750')
gdf['x'] = gdf.geometry.x
gdf['y'] = gdf.geometry.y
gdf['log_Au'] = np.log1p(gdf['Au_ppm'])

# print(" Projected to UTM")

# ============================================================================
# 3. Variogram modeling
# ============================================================================

coords = gdf[['x', 'y']].values
values = gdf['log_Au'].values

V = Variogram(coords, values, model='spherical', maxlag=50000, n_lags=20)
# print(f" Variogram: range={V.parameters[0]/1000:.1f}km, sill={V.parameters[1]:.4f}")

# ============================================================================
# 4. Ordinary Kriging baseline
# ============================================================================

x_min, x_max = gdf['x'].quantile([0.05, 0.95])
y_min, y_max = gdf['y'].quantile([0.05, 0.95])

nx, ny = 80, 80
gridx = np.linspace(x_min, x_max, nx)
gridy = np.linspace(y_min, y_max, ny)

OK = OrdinaryKriging(gdf['x'], gdf['y'], gdf['log_Au'],
                      variogram_model='spherical',
                      variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
                      verbose=False, enable_plotting=False)

z_ok, ss_ok = OK.execute('grid', gridx, gridy)
# print(" Kriging complete")

# ============================================================================
# 5. Sequential Gaussian Simulation
# ============================================================================

n_realizations = 50
simulations = []

for i in range(n_realizations):
    noise = np.random.normal(0, 0.1, len(gdf))
    perturbed = gdf['log_Au'].values + noise
    
    OK_sim = OrdinaryKriging(gdf['x'], gdf['y'], perturbed,
                              variogram_model='spherical',
                              variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
                              verbose=False, enable_plotting=False)
    
    z_sim, _ = OK_sim.execute('grid', gridx, gridy)
    simulations.append(z_sim)

sim_stack = np.stack(simulations)
# print(f" Generated {n_realizations} realizations")

# ============================================================================
# 6. Uncertainty quantification
# ============================================================================

mean_sim = np.mean(sim_stack, axis=0)
std_sim = np.std(sim_stack, axis=0)
p10_sim = np.percentile(sim_stack, 10, axis=0)
p50_sim = np.percentile(sim_stack, 50, axis=0)
p90_sim = np.percentile(sim_stack, 90, axis=0)

grade_ok = np.expm1(z_ok)
grade_mean = np.expm1(mean_sim)
grade_p10 = np.expm1(p10_sim)
grade_p90 = np.expm1(p90_sim)

print("\nResults:")
print(f"  Kriging mean: {grade_ok.mean():.3f} ppm")
print(f"  SGS mean: {grade_mean.mean():.3f} ppm")
print(f"  SGS P10: {grade_p10.mean():.3f} ppm")
print(f"  SGS P90: {grade_p90.mean():.3f} ppm")

# ============================================================================
# 7. Exceedance probability
# ============================================================================

cutoff = 0.5
prob_exceed = (sim_stack > np.log1p(cutoff)).mean(axis=0)
print(f"\nProbability(grade > {cutoff} ppm): {prob_exceed.mean():.1%}")

# ======================================================================
# Code Block 11
# ======================================================================

# Total samples: 180,294
# Geographic extent: All of Australia (7.7M km)
# Collection period: 1989-2022
# Commodities: Au, Cu, Zn, Pb, Ag, As, Mo, W, Sn, REE
# Detection limits: Au = 0.1 ppb, Cu = 1 ppm

# ======================================================================
# Code Block 12
# ======================================================================

# 'sample_id': [f'GA{i:06d}' for i in range(n_samples)],
# 'longitude': lons,
# 'latitude': lats,
# 'Au_ppm': gold_ppm,
# 'sample_type': np.random.choice(['soil', 'stream_sed', 'rock'], n_samples, p=[0.7, 0.2, 0.1])

# ======================================================================
# Code Block 13
# ======================================================================

coords, 
values, 
model='spherical',
maxlag=50000,  # 50 km max
n_lags=20,
normalize=False

# ======================================================================
# Code Block 14
# ======================================================================

gdf['x'].values,
gdf['y'].values,
gdf['log_Au'].values,
variogram_model='spherical',
variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
verbose=False,
enable_plotting=False

# ======================================================================
# Code Block 15
# ======================================================================

if (i+1) % 10 == 0:
    print(f"  Realization {i+1}/{n_realizations}...")

# ======================================================================
# Code Block 16
# ======================================================================

noise = np.random.normal(0, 0.1, len(gdf))
perturbed = gdf['log_Au'].values + noise

# ======================================================================
# Code Block 17
# ======================================================================

OK_sim = OrdinaryKriging(
    gdf['x'].values,
    gdf['y'].values,
    perturbed,
    variogram_model='spherical',
    variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
    verbose=False,
    enable_plotting=False
)

z_sim, _ = OK_sim.execute('grid', gridx, gridy)
simulations.append(z_sim)

# ======================================================================
# Code Block 18
# ======================================================================

variogram_model='spherical',
variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
# verbose=False, enable_plotting=False

# ======================================================================
# Code Block 19
# ======================================================================

noise = np.random.normal(0, 0.1, len(gdf))
perturbed = gdf['log_Au'].values + noise

OK_sim = OrdinaryKriging(gdf['x'], gdf['y'], perturbed,
                          variogram_model='spherical',
                          variogram_parameters={'sill': 0.0954, 'range': 12300, 'nugget': 0.0187},
                          verbose=False, enable_plotting=False)

z_sim, _ = OK_sim.execute('grid', gridx, gridy)
simulations.append(z_sim)
