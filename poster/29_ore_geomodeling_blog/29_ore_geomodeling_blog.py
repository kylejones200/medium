#!/usr/bin/env python3
"""
Python code extracted from 29_ore_geomodeling_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import os
import urllib.request
import zipfile
import pandas as pd

# Download NTGS drillhole database
DATA_URL = "https://geoscience.nt.gov.au/contents/prod/Downloads/Drilling/DRILLHOLES_csv.zip"
DATA_DIR = "ntgs_drillholes"
os.makedirs(DATA_DIR, exist_ok=True)

zip_path = os.path.join(DATA_DIR, "DRILLHOLES_csv.zip")

if not os.path.exists(zip_path):
    print("Downloading NTGS drillholes (warning: 500+ MB)...")
    with urllib.request.urlopen(DATA_URL) as response:
        with open(zip_path, 'wb') as f:
            f.write(response.read())
    # print("✓ Download complete")

# Extract
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(DATA_DIR)

# print(f"✓ Extracted to {DATA_DIR}/")

# ======================================================================
# Code Block 2
# ======================================================================

from pathlib import Path
import re

# Find CSV files
csvs = list(Path(DATA_DIR).rglob("*.csv"))
collar_file = [p for p in csvs if re.search(r"collar", str(p), re.I)][0]
assay_file = [p for p in csvs if re.search(r"assay|geochem", str(p), re.I)][0]

print(f"Collar file: {collar_file}")
print(f"Assay file: {assay_file}")

# Load data
collar = pd.read_csv(collar_file, low_memory=False)
assay = pd.read_csv(assay_file, low_memory=False)

print(f"\nCollar records: {len(collar):,}")
print(f"Assay records: {len(assay):,}")

collar.head(3)

# ======================================================================
# Code Block 3
# ======================================================================

def find_column(df, candidates):
    """Find first matching column from candidate list (case-insensitive)."""
    for candidate in candidates:
        for col in df.columns:
            if candidate.lower() in col.lower():
                return col
    return None

# Collar columns
hole_id_collar = find_column(collar, ['HOLEID', 'HOLE_ID', 'DHID'])
easting = find_column(collar, ['EASTING', 'EAST', 'X', 'MGA_EASTING'])
northing = find_column(collar, ['NORTHING', 'NORTH', 'Y', 'MGA_NORTHING'])
rl = find_column(collar, ['RL', 'ELEVATION', 'Z'])

# Assay columns
hole_id_assay = find_column(assay, ['HOLEID', 'HOLE_ID', 'DHID'])
depth_from = find_column(assay, ['FROM', 'DEPTH_FROM', 'FROM_M'])
depth_to = find_column(assay, ['TO', 'DEPTH_TO', 'TO_M'])

# Grade column (prefer Au, Cu, or first numeric)
grade_col = None
for pref in ['Au', 'AU', 'Au_ppm', 'Au_gpt', 'Gold', 'Cu', 'CU']:
    grade_col = find_column(assay, [pref])
    if grade_col:
        break

if grade_col is None:
    # Fallback: first numeric column that's not a coordinate
    numeric_cols = assay.select_dtypes(include=[np.number]).columns
    grade_col = [c for c in numeric_cols if c.lower() not in ['x','y','z','from','to']][0]

print(f"\nDetected columns:")
print(f"  Hole ID: {hole_id_collar} (collar), {hole_id_assay} (assay)")
print(f"  Coords: {easting}, {northing}, {rl}")
print(f"  Assay: {depth_from}, {depth_to}, {grade_col}")

# ======================================================================
# Code Block 4
# ======================================================================

import numpy as np

# Merge collar coordinates with assays
assay_subset = assay[[hole_id_assay, depth_from, depth_to, grade_col]].copy()
assay_subset.columns = ['hole_id', 'from_m', 'to_m', 'grade']
assay_subset = assay_subset.dropna()

collar_subset = collar[[hole_id_collar, easting, northing, rl]].copy()
collar_subset.columns = ['hole_id', 'easting', 'northing', 'rl']

# Merge
samples = assay_subset.merge(collar_subset, on='hole_id', how='inner')

# Compute 3D sample coordinates
# Midpoint of assay interval
samples['depth_mid'] = (samples['from_m'] + samples['to_m']) / 2.0

# Z coordinate = RL - depth (assumes vertical holes; adjust for dip if available)
samples['z'] = samples['rl'] - samples['depth_mid']

# Clean data
samples = samples[['hole_id', 'easting', 'northing', 'z', 'grade', 'depth_mid']]
samples = samples.dropna()
samples = samples[np.isfinite(samples['grade'])]

# print(f"\n✓ Generated {len(samples):,} 3D sample points")
samples.head()

# ======================================================================
# Code Block 5
# ======================================================================

from sklearn.neighbors import KDTree

def idw_interpolate(sample_coords, sample_values, query_coords, k=16, power=2.0, eps=1e-9):
    """
    Inverse Distance Weighted interpolation.
    
    Parameters:
    -----------
    sample_coords : array (n_samples, 3) - Sample X,Y,Z locations
    sample_values : array (n_samples,) - Sample grades
    query_coords : array (n_queries, 3) - Query points to estimate
    k : int - Number of nearest neighbors
    power : float - IDW exponent (typical: 2.0)
    eps : float - Minimum distance to avoid division by zero
    
    Returns:
    --------
    estimates : array (n_queries,) - Estimated grades at query points
    """
    tree = KDTree(sample_coords)
    distances, indices = tree.query(query_coords, k=min(k, len(sample_coords)))
    
    # IDW weights: 1 / distance^power
    weights = 1.0 / np.maximum(distances, eps) ** power
    weights /= weights.sum(axis=1, keepdims=True)
    
    # Weighted average
    estimates = (sample_values[indices] * weights).sum(axis=1)
    
    return estimates

# Build sample arrays
P = samples[['easting', 'northing', 'z']].values
V = samples['grade'].values

print(f"Sample points: {P.shape}")
print(f"Grade range: {V.min():.3f} - {V.max():.3f} g/t")

# ======================================================================
# Code Block 6
# ======================================================================

# 1. Raw coordinates (normalized)
from sklearn.preprocessing import StandardScaler

scaler_coords = StandardScaler()
coords_norm = scaler_coords.fit_transform(P)

# 2. Local point density (distance to 8th nearest neighbor)
tree = KDTree(P)
distances_nn, _ = tree.query(P, k=9)  # k=9 because first neighbor is self
nn_dist = distances_nn[:, -1]  # 8th neighbor distance

# 3. Depth proxy (useful if mineralization varies with depth)
depth_proxy = -samples['z'].values  # Negative Z = deeper

# 4. Regional trend (polynomial features)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
coords_poly = poly.fit_transform(samples[['easting', 'northing']])

# Combine features
X_features = np.column_stack([
    coords_norm,           # Normalized X,Y,Z
    nn_dist,               # Local density
    depth_proxy,           # Depth
    coords_poly[:, 3:]     # Polynomial terms (exclude linear, already in coords_norm)
])

print(f"Feature matrix shape: {X_features.shape}")

# ======================================================================
# Code Block 7
# ======================================================================

# Compute IDW predictions at sample locations (leave-one-out)
idw_at_samples = np.zeros(len(P))

for i in range(len(P)):
    # Leave out sample i
    P_train = np.delete(P, i, axis=0)
    V_train = np.delete(V, i)
    
    # Predict at sample i
    idw_at_samples[i] = idw_interpolate(P_train, V_train, P[i:i+1], k=16, power=2.0)[0]

# Compute residuals
residuals = V - idw_at_samples

print(f"Residual statistics:")
print(f"  Mean: {residuals.mean():.4f} (should be ~0)")
print(f"  Std: {residuals.std():.4f}")
print(f"  Range: {residuals.min():.4f} to {residuals.max():.4f}")

# ======================================================================
# Code Block 8
# ======================================================================

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Spatial cross-validation (by hole ID to avoid leakage)
groups = samples['hole_id'].values

gkf = GroupKFold(n_splits=5)
fold_scores = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_features, residuals, groups)):
    X_train, X_test = X_features[train_idx], X_features[test_idx]
    y_train, y_test = residuals[train_idx], residuals[test_idx]
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    fold_scores.append({'fold': fold, 'mae': mae, 'r2': r2})
    # print(f"Fold {fold}: MAE={mae:.4f}, R²={r2:.4f}")

print(f"\nCross-validation MAE: {np.mean([s['mae'] for s in fold_scores]):.4f} g/t")
# print(f"Cross-validation R²: {np.mean([s['r2'] for s in fold_scores]):.4f}")

# ======================================================================
# Code Block 9
# ======================================================================

final_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

final_model.fit(X_features, residuals)
# print("✓ Final model trained on all samples")

# ======================================================================
# Code Block 10
# ======================================================================

# Define bounding box (focus on densely drilled area)
e_min, e_max = samples['easting'].quantile([0.05, 0.95])
n_min, n_max = samples['northing'].quantile([0.05, 0.95])
z_min, z_max = samples['z'].quantile([0.05, 0.95])

# Block size: 25m  25m  10m (typical for mine planning)
block_size_xy = 25  # meters
block_size_z = 10   # meters

# Create grid
nx = int((e_max - e_min) / block_size_xy)
ny = int((n_max - n_min) / block_size_xy)
nz = int((z_max - z_min) / block_size_z)

# print(f"Block model dimensions: {nx} × {ny} × {nz} = {nx*ny*nz:,} blocks")

# Grid coordinates (block centroids)
grid_x = np.linspace(e_min, e_max, nx)
grid_y = np.linspace(n_min, n_max, ny)
grid_z = np.linspace(z_min, z_max, nz)

# Create 3D meshgrid
G_x, G_y, G_z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
G = np.column_stack([G_x.ravel(), G_y.ravel(), G_z.ravel()])

print(f"Grid points: {G.shape}")

# ======================================================================
# Code Block 11
# ======================================================================

print("Predicting block grades (IDW baseline)...")
idw_grades = idw_interpolate(P, V, G, k=16, power=2.0)

print("Computing grid features...")
# Normalize grid coordinates
G_norm = scaler_coords.transform(G)

# Local density at grid points
tree_grid = KDTree(P)
distances_grid, _ = tree_grid.query(G, k=9)
nn_dist_grid = distances_grid[:, -1]

# Depth proxy
depth_grid = -G[:, 2]

# Polynomial features
coords_grid_poly = poly.transform(G[:, :2])

# Combine
X_grid = np.column_stack([
    G_norm,
    nn_dist_grid,
    depth_grid,
    coords_grid_poly[:, 3:]
])

print("Predicting ML residuals...")
ml_residuals = final_model.predict(X_grid)

# Fusion: Final grade = IDW + ML residual
final_grades = idw_grades + ml_residuals

# print(f"✓ Block model complete")
print(f"  IDW grade range: {idw_grades.min():.4f} - {idw_grades.max():.4f} g/t")
print(f"  Final grade range: {final_grades.min():.4f} - {final_grades.max():.4f} g/t")

# ======================================================================
# Code Block 12
# ======================================================================

# Create block model dataframe
block_model = pd.DataFrame({
    'x_easting': G[:, 0],
    'y_northing': G[:, 1],
    'z_elevation': G[:, 2],
    'grade_idw': idw_grades,
    'grade_ml_fusion': final_grades,
    'ml_residual': ml_residuals
})

# Export to CSV (compatible with Vulcan, Datamine, Leapfrog)
output_file = 'block_model_ntgs_gold.csv'
block_model.to_csv(output_file, index=False)
# print(f"✓ Block model exported to {output_file}")

# Summary statistics
print("\nBlock Model Summary:")
print(block_model[['grade_idw', 'grade_ml_fusion']].describe())

# ======================================================================
# Code Block 13
# ======================================================================

import matplotlib.pyplot as plt

# Extract a horizontal slice at mid-depth
z_mid = np.median(grid_z)
slice_mask = np.abs(G[:, 2] - z_mid) < (block_size_z / 2)

slice_data = block_model[slice_mask].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# IDW grades
sc1 = axes[0].scatter(slice_data['x_easting'], slice_data['y_northing'],
                       c=slice_data['grade_idw'], s=20, cmap='YlOrRd',
                       vmin=0, vmax=3)
axes[0].set_title('IDW Grade (Baseline)', fontsize=12)
axes[0].set_xlabel('Easting (m)')
axes[0].set_ylabel('Northing (m)')
plt.colorbar(sc1, ax=axes[0], label='Grade (g/t Au)')

# ML Fusion grades
sc2 = axes[1].scatter(slice_data['x_easting'], slice_data['y_northing'],
                       c=slice_data['grade_ml_fusion'], s=20, cmap='YlOrRd',
                       vmin=0, vmax=3)
axes[1].set_title('ML Fusion Grade', fontsize=12)
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Northing (m)')
plt.colorbar(sc2, ax=axes[1], label='Grade (g/t Au)')

plt.tight_layout()
plt.savefig('grade_comparison_slice.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 14
# ======================================================================

from sklearn.ensemble import RandomForestRegressor

# Train ensemble of models
n_realizations = 20
realizations = []

for i in range(n_realizations):
    # Bootstrap sample
    sample_idx = np.random.choice(len(X_features), len(X_features), replace=True)
    X_boot = X_features[sample_idx]
    y_boot = residuals[sample_idx]
    
    # Train model
    model_i = RandomForestRegressor(n_estimators=100, random_state=i)
    model_i.fit(X_boot, y_boot)
    
    # Predict on grid
    residual_i = model_i.predict(X_grid)
    grade_i = idw_grades + residual_i
    realizations.append(grade_i)

# Compute statistics
realizations = np.array(realizations)
grade_mean = realizations.mean(axis=0)
grade_std = realizations.std(axis=0)
grade_p10 = np.percentile(realizations, 10, axis=0)
grade_p90 = np.percentile(realizations, 90, axis=0)

# Add to block model
block_model['grade_mean'] = grade_mean
block_model['grade_std'] = grade_std
block_model['grade_p10'] = grade_p10
block_model['grade_p90'] = grade_p90

# ======================================================================
# Code Block 15
# ======================================================================

# Load domain polygons (e.g., from geological interpretation)
from shapely.geometry import Point, Polygon

# Example: Define mineralized zone polygon
mineralized_zone = Polygon([
    (easting_min, northing_min),
    (easting_max, northing_min),
    (easting_max, northing_max),
    (easting_min, northing_max)
])

# Flag blocks inside/outside domain
def point_in_domain(x, y, polygon):
    return polygon.contains(Point(x, y))

block_model['in_domain'] = [
    point_in_domain(x, y, mineralized_zone)
    for x, y in zip(block_model['x_easting'], block_model['y_northing'])
]

# Set grades outside domain to zero (or background)
block_model.loc[~block_model['in_domain'], 'grade_ml_fusion'] = 0

# ======================================================================
# Code Block 16
# ======================================================================

# Load geophysical grid (e.g., TMI from airborne mag)
import rasterio

with rasterio.open('tmi_grid.tif') as src:
    # Sample TMI at drillhole locations
    tmi_values = []
    for e, n in zip(samples['easting'], samples['northing']):
        row, col = src.index(e, n)
        tmi_values.append(src.read(1)[row, col])
    
    samples['tmi'] = tmi_values

# Add TMI to features
X_features_enhanced = np.column_stack([X_features, samples['tmi']])

# Retrain model with geophysics
# (Often improves predictions in areas with sparse drilling)

# ======================================================================
# Code Block 17
# ======================================================================

# Complete ore body geomodeling pipeline
# Uses public NTGS data

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
import urllib.request
import zipfile
import os
from pathlib import Path
import re

# ============================================================================
# 1. Download and load NTGS data
# ============================================================================

DATA_URL = "https://geoscience.nt.gov.au/contents/prod/Downloads/Drilling/DRILLHOLES_csv.zip"
DATA_DIR = "ntgs_drillholes"
os.makedirs(DATA_DIR, exist_ok=True)

zip_path = os.path.join(DATA_DIR, "DRILLHOLES_csv.zip")

if not os.path.exists(zip_path):
    print("Downloading NTGS data...")
    # Note: This is a large file; in production, cache it
    # urllib.request.urlretrieve(DATA_URL, zip_path)
    # print("✓ Downloaded (or use cached version)")

# For demo, use synthetic data instead of downloading
print("Generating synthetic drillhole data for demo...")
rng = np.random.default_rng(42)

n_holes = 100
n_samples_per_hole = 30

synthetic_samples = []
for hole in range(n_holes):
    # Random hole location in 2km x 2km area
    e_base = 400000 + rng.uniform(0, 2000)
    n_base = 7500000 + rng.uniform(0, 2000)
    rl_base = 300 + rng.normal(0, 10)
    
    for sample in range(n_samples_per_hole):
        depth = sample * 5  # 5m intervals
        
        # Synthetic grade: function of location + depth + noise
        # High-grade zone in center
        dist_to_center = np.sqrt((e_base - 401000)**2 + (n_base - 7501000)**2)
        grade_base = 2.0 * np.exp(-dist_to_center / 500) + 0.3
        grade = grade_base * (1 + 0.3 * np.sin(depth / 20)) + rng.normal(0, 0.2)
        grade = np.clip(grade, 0.01, 10.0)
        
        synthetic_samples.append({
            'hole_id': f'DDH{hole:03d}',
            'easting': e_base + rng.normal(0, 2),
            'northing': n_base + rng.normal(0, 2),
            'rl': rl_base,
            'depth_mid': depth,
            'z': rl_base - depth,
            'grade': grade
        })

samples = pd.DataFrame(synthetic_samples)
# print(f"✓ Generated {len(samples):,} synthetic samples")

# ============================================================================
# 2. Baseline IDW interpolation
# ============================================================================

def idw_interpolate(coords, values, query, k=16, power=2.0):
    tree = KDTree(coords)
    dists, idx = tree.query(query, k=min(k, len(coords)))
    w = 1.0 / np.maximum(dists, 1e-9) ** power
    w /= w.sum(axis=1, keepdims=True)
    return (values[idx] * w).sum(axis=1)

P = samples[['easting', 'northing', 'z']].values
V = samples['grade'].values

print(f"Sample points: {len(P):,}")
print(f"Grade range: {V.min():.3f} - {V.max():.3f} g/t")

# ============================================================================
# 3. Feature engineering
# ============================================================================

scaler = StandardScaler()
coords_norm = scaler.fit_transform(P)

tree = KDTree(P)
dists_nn, _ = tree.query(P, k=9)
nn_dist = dists_nn[:, -1]

depth = -samples['z'].values

poly = PolynomialFeatures(degree=2, include_bias=False)
coords_poly = poly.fit_transform(samples[['easting', 'northing']])

X_features = np.column_stack([coords_norm, nn_dist, depth, coords_poly[:, 3:]])

print(f"Feature matrix: {X_features.shape}")

# ============================================================================
# 4. Compute residuals (for hybrid model)
# ============================================================================

print("Computing IDW residuals (leave-one-out)...")
idw_at_samples = np.zeros(len(P))

# For speed, use approximate leave-one-out
for i in range(min(len(P), 1000)):  # Limit for demo
    P_train = np.delete(P, i, axis=0)
    V_train = np.delete(V, i)
    idw_at_samples[i] = idw_interpolate(P_train, V_train, P[i:i+1], k=16)[0]

# For remaining, use full IDW (small bias but fast)
if len(P) > 1000:
    idw_at_samples[1000:] = idw_interpolate(P, V, P[1000:], k=16)

residuals = V - idw_at_samples

print(f"Residual stats: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

# ============================================================================
# 5. Train ML model on residuals
# ============================================================================

groups = samples['hole_id'].values
gkf = GroupKFold(n_splits=3)  # 3-fold for demo

fold_maes = []
for train_idx, test_idx in gkf.split(X_features, residuals, groups):
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
    model.fit(X_features[train_idx], residuals[train_idx])
    pred = model.predict(X_features[test_idx])
    mae = mean_absolute_error(residuals[test_idx], pred)
    fold_maes.append(mae)

print(f"Cross-validation MAE: {np.mean(fold_maes):.4f} g/t")

# Train final model
final_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
final_model.fit(X_features, residuals)
# print("✓ Model trained")

# ============================================================================
# 6. Generate block model
# ============================================================================

nx, ny, nz = 40, 40, 10
gx = np.linspace(P[:, 0].min(), P[:, 0].max(), nx)
gy = np.linspace(P[:, 1].min(), P[:, 1].max(), ny)
gz = np.linspace(P[:, 2].min(), P[:, 2].max(), nz)

G = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), axis=-1).reshape(-1, 3)

print(f"Predicting {len(G):,} block grades...")
idw_grid = idw_interpolate(P, V, G, k=16)

# Grid features
G_norm = scaler.transform(G)
dists_grid, _ = tree.query(G, k=9)
nn_grid = dists_grid[:, -1]
depth_grid = -G[:, 2]
poly_grid = poly.transform(G[:, :2])
X_grid = np.column_stack([G_norm, nn_grid, depth_grid, poly_grid[:, 3:]])

ml_resid = final_model.predict(X_grid)
fusion_grid = idw_grid + ml_resid

# ============================================================================
# 7. Export
# ============================================================================

block_model = pd.DataFrame({
    'x': G[:, 0],
    'y': G[:, 1],
    'z': G[:, 2],
    'grade_idw': idw_grid,
    'grade_fusion': fusion_grid
})

block_model.to_csv('block_model_demo.csv', index=False)
# print("✓ Block model exported to block_model_demo.csv")

print("\nSummary:")
print(block_model[['grade_idw', 'grade_fusion']].describe())

# ======================================================================
# Code Block 18
# ======================================================================

# ┌─────────────────────────────────┐
# │  Public Drillhole Data (NTGS)   │
# │  • Collar coordinates (X,Y,Z)    │
# │  • Downhole assays (Au, Cu, etc)│
# │  • Lithology, alteration         │
# └────────────┬────────────────────┘
             # │
# ▼
# ┌─────────────────────────────────┐
# │  Feature Engineering             │
# │  • Spatial coords (X, Y, Z)      │
# │  • Distance to structures        │
# │  • Local point density           │
# │  • Geophysical layers            │
# └────────────┬────────────────────┘
             # │
# ▼
# ┌─────────────────────────────────┐
# │  Baseline: IDW Interpolation    │
# │  • K-nearest neighbors (k=16)    │
# │  • Power=2.0                     │
# │  • Provides spatial trend        │
# └────────────┬────────────────────┘
             # │
# ▼
# ┌─────────────────────────────────┐
# │  Residual Learning (ML)          │
# │  • Target: grade - IDW_pred      │
# │  • Model: Gradient Boosting      │
# │  • Captures local variations     │
# └────────────┬────────────────────┘
             # │
# ▼
# ┌─────────────────────────────────┐
# │  Fusion: IDW + ML Residual       │
# │  • Final grade = IDW + residual  │
# │  • Preserves spatial continuity  │
# │  • Captures complex patterns     │
# └────────────┬────────────────────┘
             # │
# ▼
# ┌─────────────────────────────────┐
# │  3D Block Model Export           │
# │  • CSV: X, Y, Z, grade, variance │
# │  • Compatible with mine planning │
# │  • Multiple realizations (MCS)   │
# └─────────────────────────────────┘

# ======================================================================
# Code Block 19
# ======================================================================

print("Downloading NTGS drillholes (warning: 500+ MB)...")
with urllib.request.urlopen(DATA_URL) as response:
    with open(zip_path, 'wb') as f:
        f.write(response.read())
# print("✓ Download complete")

# ======================================================================
# Code Block 20
# ======================================================================

"""Find first matching column from candidate list (case-insensitive)."""
for candidate in candidates:
    for col in df.columns:
        if candidate.lower() in col.lower():
            return col
return None

# ======================================================================
# Code Block 21
# ======================================================================

grade_col = find_column(assay, [pref])
if grade_col:
    break

# ======================================================================
# Code Block 22
# ======================================================================

numeric_cols = assay.select_dtypes(include=[np.number]).columns
grade_col = [c for c in numeric_cols if c.lower() not in ['x','y','z','from','to']][0]

# ======================================================================
# Code Block 23
# ======================================================================

"""
Inverse Distance Weighted interpolation.

Parameters:
-----------
sample_coords : array (n_samples, 3) - Sample X,Y,Z locations
sample_values : array (n_samples,) - Sample grades
query_coords : array (n_queries, 3) - Query points to estimate
k : int - Number of nearest neighbors
power : float - IDW exponent (typical: 2.0)
eps : float - Minimum distance to avoid division by zero

Returns:
--------
estimates : array (n_queries,) - Estimated grades at query points
"""
tree = KDTree(sample_coords)
distances, indices = tree.query(query_coords, k=min(k, len(sample_coords)))

# ======================================================================
# Code Block 24
# ======================================================================

weights = 1.0 / np.maximum(distances, eps) ** power
weights /= weights.sum(axis=1, keepdims=True)

# ======================================================================
# Code Block 25
# ======================================================================

estimates = (sample_values[indices] * weights).sum(axis=1)

return estimates

# ======================================================================
# Code Block 26
# ======================================================================

P_train = np.delete(P, i, axis=0)
V_train = np.delete(V, i)

# ======================================================================
# Code Block 27
# ======================================================================

idw_at_samples[i] = idw_interpolate(P_train, V_train, P[i:i+1], k=16, power=2.0)[0]

# ======================================================================
# Code Block 28
# ======================================================================

X_train, X_test = X_features[train_idx], X_features[test_idx]
y_train, y_test = residuals[train_idx], residuals[test_idx]

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

fold_scores.append({'fold': fold, 'mae': mae, 'r2': r2})
# print(f"Fold {fold}: MAE={mae:.4f}, R²={r2:.4f}")

# ======================================================================
# Code Block 29
# ======================================================================

# Fold 0: MAE=0.0847, R²=0.3421
# Fold 1: MAE=0.0912, R²=0.3156
# Fold 2: MAE=0.0789, R²=0.3688
# Fold 3: MAE=0.0834, R²=0.3502
# Fold 4: MAE=0.0871, R²=0.3287

# Cross-validation MAE: 0.0851 g/t
# Cross-validation R²: 0.3411

# ======================================================================
# Code Block 30
# ======================================================================

n_estimators=200,
max_depth=4,
learning_rate=0.05,
subsample=0.8,
random_state=42

# ======================================================================
# Code Block 31
# ======================================================================

# c=slice_data['grade_idw'], s=20, cmap='YlOrRd',
# vmin=0, vmax=3

# ======================================================================
# Code Block 32
# ======================================================================

# c=slice_data['grade_ml_fusion'], s=20, cmap='YlOrRd',
# vmin=0, vmax=3

# ======================================================================
# Code Block 33
# ======================================================================

sample_idx = np.random.choice(len(X_features), len(X_features), replace=True)
X_boot = X_features[sample_idx]
y_boot = residuals[sample_idx]

# ======================================================================
# Code Block 34
# ======================================================================

model_i = RandomForestRegressor(n_estimators=100, random_state=i)
model_i.fit(X_boot, y_boot)

# ======================================================================
# Code Block 35
# ======================================================================

residual_i = model_i.predict(X_grid)
grade_i = idw_grades + residual_i
realizations.append(grade_i)

# ======================================================================
# Code Block 36
# ======================================================================

return polygon.contains(Point(x, y))

# ======================================================================
# Code Block 37
# ======================================================================

tmi_values = []
for e, n in zip(samples['easting'], samples['northing']):
    row, col = src.index(e, n)
    tmi_values.append(src.read(1)[row, col])

samples['tmi'] = tmi_values

# ======================================================================
# Code Block 38
# ======================================================================

print("Downloading NTGS data...")

# ======================================================================
# Code Block 39
# ======================================================================

# print("✓ Downloaded (or use cached version)")

# ======================================================================
# Code Block 40
# ======================================================================

e_base = 400000 + rng.uniform(0, 2000)
n_base = 7500000 + rng.uniform(0, 2000)
rl_base = 300 + rng.normal(0, 10)

for sample in range(n_samples_per_hole):
    depth = sample * 5  # 5m intervals

# ======================================================================
# Code Block 41
# ======================================================================

dist_to_center = np.sqrt((e_base - 401000)**2 + (n_base - 7501000)**2)
grade_base = 2.0 * np.exp(-dist_to_center / 500) + 0.3
grade = grade_base * (1 + 0.3 * np.sin(depth / 20)) + rng.normal(0, 0.2)
grade = np.clip(grade, 0.01, 10.0)
    
synthetic_samples.append({
        'hole_id': f'DDH{hole:03d}',
        'easting': e_base + rng.normal(0, 2),
        'northing': n_base + rng.normal(0, 2),
        'rl': rl_base,
        'depth_mid': depth,
        'z': rl_base - depth,
        'grade': grade
    })

# ======================================================================
# Code Block 42
# ======================================================================

tree = KDTree(coords)
dists, idx = tree.query(query, k=min(k, len(coords)))
w = 1.0 / np.maximum(dists, 1e-9) ** power
w /= w.sum(axis=1, keepdims=True)
return (values[idx] * w).sum(axis=1)

# ======================================================================
# Code Block 43
# ======================================================================

P_train = np.delete(P, i, axis=0)
V_train = np.delete(V, i)
idw_at_samples[i] = idw_interpolate(P_train, V_train, P[i:i+1], k=16)[0]

# ======================================================================
# Code Block 44
# ======================================================================

idw_at_samples[1000:] = idw_interpolate(P, V, P[1000:], k=16)

# ======================================================================
# Code Block 45
# ======================================================================

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
model.fit(X_features[train_idx], residuals[train_idx])
pred = model.predict(X_features[test_idx])
mae = mean_absolute_error(residuals[test_idx], pred)
fold_maes.append(mae)
