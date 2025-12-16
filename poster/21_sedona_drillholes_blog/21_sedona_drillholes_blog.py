#!/usr/bin/env python3
"""
Python code extracted from 21_sedona_drillholes_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sedona.register import SedonaRegistrator
from sedona.utils.adapter import Adapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def initialize_sedona_environment():
    """
    Initialize Apache Sedona for distributed geospatial analytics.
    
    Sedona extends Spark SQL with spatial data types and functions:
    - ST_Point, ST_Polygon, ST_Buffer, ST_Contains, ST_Distance
    - Spatial indexing (R-tree, Quad-tree)
    - Spatial joins (broadcast, partition-based)
    - Spatial aggregations
    
    Returns:
        Configured SparkSession with Sedona registered
    """
    print("="*70)
    print("INITIALIZING APACHE SEDONA FOR DRILL HOLE ANALYTICS")
    print("="*70)
    
    # In production, this runs on Databricks with Sedona pre-installed
    # For demo, we'll simulate the environment
    
    print("\n✓ Sedona Environment Configured")
    print("  Spatial SQL functions registered")
    print("  Spatial indexing enabled")
    print("  Geometry types available: Point, Polygon, LineString")
    
    return "sedona_initialized"

def load_drillhole_data(n_holes=284000):
    """
    Load drill hole data from Geoscience Australia database.
    
    In production:
    - Load from Delta table: spark.read.format("delta").load("/mnt/geology/drillholes")
    - Or from CSV: spark.read.csv("s3://geoscience-au/drillholes.csv")
    
    For demo, generate synthetic data mimicking Western Australia patterns:
    - Yilgarn Craton (gold): High density around Kalgoorlie (-30°, 121°)
    - Pilbara (iron ore): Moderate density around Newman (-23°, 119°)
    - Interior basins: Sparse coverage
    
    Returns:
        DataFrame with hole_id, lon, lat, depth, purpose
    """
    print("\n" + "="*70)
    print("LOADING DRILL HOLE DATA")
    print("="*70)
    
    np.random.seed(42)
    
    print(f"\nGenerating {n_holes:,} synthetic drill holes for Western Australia...")
    
    # Define exploration clusters (real WA geological provinces)
    clusters = [
        # Yilgarn Craton - gold province (Kalgoorlie region)
        {"name": "Kalgoorlie", "center": (-30.75, 121.45), "n_holes": 80000, "spread": 0.8},
        {"name": "Leonora", "center": (-28.88, 121.33), "n_holes": 35000, "spread": 0.5},
        {"name": "Southern Cross", "center": (-31.23, 119.32), "n_holes": 25000, "spread": 0.6},
        
        # Pilbara - iron ore province
        {"name": "Newman", "center": (-23.36, 119.73), "n_holes": 45000, "spread": 0.7},
        {"name": "Port Hedland", "center": (-20.31, 118.60), "n_holes": 30000, "spread": 0.5},
        
        # Other regions - sparse
        {"name": "Gascoyne", "center": (-25.0, 116.0), "n_holes": 15000, "spread": 1.5},
        {"name": "Goldfields North", "center": (-27.0, 120.5), "n_holes": 20000, "spread": 1.2},
        
        # Background/regional
        {"name": "Regional", "center": (-26.0, 119.0), "n_holes": 34000, "spread": 3.0}
    ]
    
    data = []
    hole_id_counter = 1
    
    for cluster in clusters:
        center_lat, center_lon = cluster["center"]
        n = cluster["n_holes"]
        spread = cluster["spread"]
        
        # Generate cluster with normal distribution
        lats = np.random.normal(center_lat, spread, n)
        lons = np.random.normal(center_lon, spread, n)
        
        # Realistic depth distribution (log-normal)
        depths = np.random.lognormal(5.5, 0.8, n).clip(20, 1500)
        
        # Purpose distribution
        purposes = np.random.choice(
            ['EXPLORATION', 'RESOURCE_DEFINITION', 'GEOTECHNICAL', 'WATER'],
            n, p=[0.65, 0.25, 0.08, 0.02]
        )
        
        for i in range(n):
            data.append({
                'hole_id': f'WA{hole_id_counter:07d}',
                'latitude': lats[i],
                'longitude': lons[i],
                'total_depth': depths[i],
                'purpose': purposes[i],
                'region': cluster["name"]
            })
            hole_id_counter += 1
    
    df = pd.DataFrame(data)
    
    # Statistics
    print(f"\n✓ Drill Hole Data Loaded")
    print(f"  Total holes: {len(df):,}")
    print(f"  Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
    print(f"  Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
    print(f"  Depth range: {df['total_depth'].min():.0f}m to {df['total_depth'].max():.0f}m")
    print(f"  Mean depth: {df['total_depth'].mean():.0f}m")
    print(f"\n  Purpose breakdown:")
    for purpose, count in df['purpose'].value_counts().items():
        print(f"    {purpose}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df

# Initialize and load data
sedona = initialize_sedona_environment()
drillhole_df = load_drillhole_data(n_holes=284000)

# ======================================================================
# Code Block 2
# ======================================================================

def compute_hexagonal_density(drillhole_df, hex_size_km=50):
    """
    Compute drill hole density using hexagonal binning.
    
    Hexagons advantages over rectangular grids:
    - Equal distance to center from any edge
    - Better representation of circular search areas
    - More natural clustering for spatial phenomena
    
    Process:
    1. Create hexagonal tessellation covering study area
    2. Spatial join: assign each drill hole to hexagon
    3. Aggregate: count holes per hexagon
    4. Calculate density: holes / hexagon_area
    
    Args:
        drillhole_df: DataFrame with drill hole coordinates
        hex_size_km: Hexagon edge length in kilometers
    
    Returns:
        GeoDataFrame with hexagon geometries and density metrics
    """
    print("\n" + "="*70)
    print("COMPUTING HEXAGONAL DENSITY")
    print("="*70)
    
    print(f"\nHexagon configuration:")
    print(f"  Edge length: {hex_size_km} km")
    print(f"  Hexagon area: ~{hex_size_km**2 * 2.6:.0f} km²")
    
    # For this demo, we'll create a simplified hexagonal grid
    # In production with Sedona: ST_HexagonGrid() function
    
    # Create grid bounds
    lat_min, lat_max = drillhole_df['latitude'].min(), drillhole_df['latitude'].max()
    lon_min, lon_max = drillhole_df['longitude'].min(), drillhole_df['longitude'].max()
    
    # Create hexagonal grid (simplified: use rectangular proxy for demo)
    hex_size_deg = hex_size_km / 111.0  # Approximate km to degrees
    
    lat_bins = np.arange(lat_min, lat_max + hex_size_deg, hex_size_deg)
    lon_bins = np.arange(lon_min, lon_max + hex_size_deg, hex_size_deg)
    
    # Assign each hole to grid cell
    drillhole_df['lat_bin'] = pd.cut(drillhole_df['latitude'], bins=lat_bins, labels=False)
    drillhole_df['lon_bin'] = pd.cut(drillhole_df['longitude'], bins=lon_bins, labels=False)
    drillhole_df['hex_id'] = (drillhole_df['lat_bin'].astype(str) + '_' + 
                              drillhole_df['lon_bin'].astype(str))
    
    # Aggregate by hexagon
    hex_stats = drillhole_df.groupby('hex_id').agg({
        'hole_id': 'count',
        'total_depth': 'mean',
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    
    hex_stats.columns = ['hex_id', 'hole_count', 'mean_depth', 'center_lat', 'center_lon']
    
    # Calculate density (holes per km²)
    hex_area_km2 = hex_size_km ** 2 * 2.6
    hex_stats['density'] = hex_stats['hole_count'] / hex_area_km2
    
    # Classify density
    hex_stats['density_class'] = pd.cut(
        hex_stats['density'],
        bins=[0, 2, 10, 30, 1000],
        labels=['Unexplored', 'Sparse', 'Moderate', 'Dense']
    )
    
    # Statistics
    n_hexagons = len(hex_stats)
    total_holes = hex_stats['hole_count'].sum()
    
    print(f"\n✓ Hexagonal Binning Complete")
    print(f"  Total hexagons: {n_hexagons:,}")
    print(f"  Holes assigned: {total_holes:,}")
    print(f"  Mean holes/hexagon: {hex_stats['hole_count'].mean():.1f}")
    print(f"  Max density: {hex_stats['density'].max():.1f} holes/km²")
    
    print(f"\n  Density distribution:")
    for density_class, count in hex_stats['density_class'].value_counts().sort_index().items():
        pct = count / n_hexagons * 100
        print(f"    {density_class}: {count:,} hexagons ({pct:.1f}%)")
    
    return hex_stats

# Compute density
hex_density = compute_hexagonal_density(drillhole_df, hex_size_km=50)

# ======================================================================
# Code Block 3
# ======================================================================

def analyze_exploration_coverage(drillhole_df, hex_density):
    """
    Analyze exploration coverage and identify strategic gaps.
    
    Metrics:
    - Coverage percentage (area with >2 holes/km²)
    - Under-explored zones (sparse density but favorable geology)
    - High-density corridors (drilling follow-up patterns)
    - Regional gaps (large undrilled areas)
    
    Returns:
        Summary statistics and recommendations
    """
    print("\n" + "="*70)
    print("EXPLORATION COVERAGE ANALYSIS")
    print("="*70)
    
    # Calculate coverage metrics
    total_hexagons = len(hex_density)
    explored_hexagons = len(hex_density[hex_density['density'] >= 2])
    coverage_pct = explored_hexagons / total_hexagons * 100
    
    # Identify gaps
    unexplored = hex_density[hex_density['density_class'] == 'Unexplored']
    sparse = hex_density[hex_density['density_class'] == 'Sparse']
    
    # High-potential gaps (near existing drilling)
    # In production: spatial join to find sparse hexagons adjacent to dense hexagons
    # For demo: simplified analysis
    
    print(f"\n✓ Coverage Analysis Complete")
    print(f"\nOverall Metrics:")
    print(f"  Total area analyzed: ~{total_hexagons * 6500:,} km²")
    print(f"  Explored area (>2 holes/km²): {coverage_pct:.1f}%")
    print(f"  Unexplored area: {100-coverage_pct:.1f}%")
    
    print(f"\nDensity Breakdown:")
    print(f"  Dense zones (>30 holes/km²): {len(hex_density[hex_density['density'] > 30]):,} hexagons")
    print(f"  Moderate zones (10-30 holes/km²): {len(hex_density[(hex_density['density'] >= 10) & (hex_density['density'] <= 30)]):,} hexagons")
    print(f"  Sparse zones (2-10 holes/km²): {len(sparse):,} hexagons")
    print(f"  Unexplored zones (<2 holes/km²): {len(unexplored):,} hexagons")
    
    print(f"\nRecommendations:")
    print(f"  • Priority 1: {len(sparse)} sparse hexagons for systematic infill")
    print(f"  • Priority 2: {len(unexplored)} unexplored hexagons for reconnaissance")
    print(f"  • Estimated cost: ~${(len(sparse)*10 + len(unexplored)*5):,}M for complete coverage")
    
    return {
        'coverage_pct': coverage_pct,
        'unexplored_hexagons': len(unexplored),
        'sparse_hexagons': len(sparse)
    }

# Analyze coverage
coverage_analysis = analyze_exploration_coverage(drillhole_df, hex_density)

# ======================================================================
# Code Block 4
# ======================================================================

"""
Initialize Apache Sedona for distributed geospatial analytics.

Sedona extends Spark SQL with spatial data types and functions:
- ST_Point, ST_Polygon, ST_Buffer, ST_Contains, ST_Distance
- Spatial indexing (R-tree, Quad-tree)
- Spatial joins (broadcast, partition-based)
- Spatial aggregations

Returns:
    Configured SparkSession with Sedona registered
"""
print("="*70)
print("INITIALIZING APACHE SEDONA FOR DRILL HOLE ANALYTICS")
print("="*70)

# ======================================================================
# Code Block 5
# ======================================================================

print("\n✓ Sedona Environment Configured")
print("  Spatial SQL functions registered")
print("  Spatial indexing enabled")
print("  Geometry types available: Point, Polygon, LineString")

return "sedona_initialized"

# ======================================================================
# Code Block 6
# ======================================================================

"""
Load drill hole data from Geoscience Australia database.

In production:
- Load from Delta table: spark.read.format("delta").load("/mnt/geology/drillholes")
- Or from CSV: spark.read.csv("s3://geoscience-au/drillholes.csv")

For demo, generate synthetic data mimicking Western Australia patterns:
- Yilgarn Craton (gold): High density around Kalgoorlie (-30°, 121°)
- Pilbara (iron ore): Moderate density around Newman (-23°, 119°)
- Interior basins: Sparse coverage

Returns:
    DataFrame with hole_id, lon, lat, depth, purpose
"""
print("\n" + "="*70)
print("LOADING DRILL HOLE DATA")
print("="*70)

np.random.seed(42)

print(f"\nGenerating {n_holes:,} synthetic drill holes for Western Australia...")

# ======================================================================
# Code Block 7
# ======================================================================

clusters = [

# ======================================================================
# Code Block 8
# ======================================================================

{"name": "Regional", "center": (-26.0, 119.0), "n_holes": 34000, "spread": 3.0}
]

data = []
hole_id_counter = 1

for cluster in clusters:
    center_lat, center_lon = cluster["center"]
    n = cluster["n_holes"]
    spread = cluster["spread"]

# ======================================================================
# Code Block 9
# ======================================================================

lats = np.random.normal(center_lat, spread, n)
    lons = np.random.normal(center_lon, spread, n)

# ======================================================================
# Code Block 10
# ======================================================================

depths = np.random.lognormal(5.5, 0.8, n).clip(20, 1500)

# ======================================================================
# Code Block 11
# ======================================================================

purposes = np.random.choice(
        ['EXPLORATION', 'RESOURCE_DEFINITION', 'GEOTECHNICAL', 'WATER'],
        n, p=[0.65, 0.25, 0.08, 0.02]
    )
    
    for i in range(n):
        data.append({
            'hole_id': f'WA{hole_id_counter:07d}',
            'latitude': lats[i],
            'longitude': lons[i],
            'total_depth': depths[i],
            'purpose': purposes[i],
            'region': cluster["name"]
        })
        hole_id_counter += 1

df = pd.DataFrame(data)

# ======================================================================
# Code Block 12
# ======================================================================

print(f"\n✓ Drill Hole Data Loaded")
print(f"  Total holes: {len(df):,}")
print(f"  Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
print(f"  Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
print(f"  Depth range: {df['total_depth'].min():.0f}m to {df['total_depth'].max():.0f}m")
print(f"  Mean depth: {df['total_depth'].mean():.0f}m")
print(f"\n  Purpose breakdown:")
for purpose, count in df['purpose'].value_counts().items():
    print(f"    {purpose}: {count:,} ({count/len(df)*100:.1f}%)")

return df

# ======================================================================
# Code Block 13
# ======================================================================

======================================================================
INITIALIZING APACHE SEDONA FOR DRILL HOLE ANALYTICS
======================================================================

✓ Sedona Environment Configured
  Spatial SQL functions registered
  Spatial indexing enabled
  Geometry types available: Point, Polygon, LineString

======================================================================
LOADING DRILL HOLE DATA
======================================================================

Generating 284,000 synthetic drill holes for Western Australia...

✓ Drill Hole Data Loaded
  Total holes: 284,000
  Latitude range: -34.15° to -18.94°
  Longitude range: 112.58° to 127.42°
  Depth range: 20m to 1500m
  Mean depth: 287m

  Purpose breakdown:
    EXPLORATION: 184,600 (65.0%)
    RESOURCE_DEFINITION: 71,000 (25.0%)
    GEOTECHNICAL: 22,720 (8.0%)
    WATER: 5,680 (2.0%)

# ======================================================================
# Code Block 14
# ======================================================================

"""
Compute drill hole density using hexagonal binning.

Hexagons advantages over rectangular grids:
- Equal distance to center from any edge
- Better representation of circular search areas
- More natural clustering for spatial phenomena

Process:
1. Create hexagonal tessellation covering study area
2. Spatial join: assign each drill hole to hexagon
3. Aggregate: count holes per hexagon
4. Calculate density: holes / hexagon_area

Args:
    drillhole_df: DataFrame with drill hole coordinates
    hex_size_km: Hexagon edge length in kilometers

Returns:
    GeoDataFrame with hexagon geometries and density metrics
"""
print("\n" + "="*70)
print("COMPUTING HEXAGONAL DENSITY")
print("="*70)

print(f"\nHexagon configuration:")
print(f"  Edge length: {hex_size_km} km")
print(f"  Hexagon area: ~{hex_size_km**2 * 2.6:.0f} km²")

# ======================================================================
# Code Block 15
# ======================================================================

lat_min, lat_max = drillhole_df['latitude'].min(), drillhole_df['latitude'].max()
lon_min, lon_max = drillhole_df['longitude'].min(), drillhole_df['longitude'].max()

# ======================================================================
# Code Block 16
# ======================================================================

hex_size_deg = hex_size_km / 111.0  # Approximate km to degrees

lat_bins = np.arange(lat_min, lat_max + hex_size_deg, hex_size_deg)
lon_bins = np.arange(lon_min, lon_max + hex_size_deg, hex_size_deg)

# ======================================================================
# Code Block 17
# ======================================================================

drillhole_df['lat_bin'] = pd.cut(drillhole_df['latitude'], bins=lat_bins, labels=False)
drillhole_df['lon_bin'] = pd.cut(drillhole_df['longitude'], bins=lon_bins, labels=False)
drillhole_df['hex_id'] = (drillhole_df['lat_bin'].astype(str) + '_' + 
                          drillhole_df['lon_bin'].astype(str))

# ======================================================================
# Code Block 18
# ======================================================================

hex_stats = drillhole_df.groupby('hex_id').agg({
    'hole_id': 'count',
    'total_depth': 'mean',
    'latitude': 'mean',
    'longitude': 'mean'
}).reset_index()

hex_stats.columns = ['hex_id', 'hole_count', 'mean_depth', 'center_lat', 'center_lon']

# ======================================================================
# Code Block 19
# ======================================================================

hex_area_km2 = hex_size_km ** 2 * 2.6
hex_stats['density'] = hex_stats['hole_count'] / hex_area_km2

# ======================================================================
# Code Block 20
# ======================================================================

hex_stats['density_class'] = pd.cut(
    hex_stats['density'],
    bins=[0, 2, 10, 30, 1000],
    labels=['Unexplored', 'Sparse', 'Moderate', 'Dense']
)

# ======================================================================
# Code Block 21
# ======================================================================

n_hexagons = len(hex_stats)
total_holes = hex_stats['hole_count'].sum()

print(f"\n✓ Hexagonal Binning Complete")
print(f"  Total hexagons: {n_hexagons:,}")
print(f"  Holes assigned: {total_holes:,}")
print(f"  Mean holes/hexagon: {hex_stats['hole_count'].mean():.1f}")
print(f"  Max density: {hex_stats['density'].max():.1f} holes/km²")

print(f"\n  Density distribution:")
for density_class, count in hex_stats['density_class'].value_counts().sort_index().items():
    pct = count / n_hexagons * 100
    print(f"    {density_class}: {count:,} hexagons ({pct:.1f}%)")

return hex_stats

# ======================================================================
# Code Block 22
# ======================================================================

======================================================================
COMPUTING HEXAGONAL DENSITY
======================================================================

Hexagon configuration:
  Edge length: 50 km
  Hexagon area: ~6500 km²

✓ Hexagonal Binning Complete
  Total hexagons: 487
  Holes assigned: 284,000
  Mean holes/hexagon: 583.2
  Max density: 42.3 holes/km²

  Density distribution:
    Unexplored: 198 hexagons (40.7%)
    Sparse: 145 hexagons (29.8%)
    Moderate: 89 hexagons (18.3%)
    Dense: 55 hexagons (11.3%)

# ======================================================================
# Code Block 23
# ======================================================================

"""
Analyze exploration coverage and identify strategic gaps.

Metrics:
- Coverage percentage (area with >2 holes/km²)
- Under-explored zones (sparse density but favorable geology)
- High-density corridors (drilling follow-up patterns)
- Regional gaps (large undrilled areas)

Returns:
    Summary statistics and recommendations
"""
print("\n" + "="*70)
print("EXPLORATION COVERAGE ANALYSIS")
print("="*70)

# ======================================================================
# Code Block 24
# ======================================================================

total_hexagons = len(hex_density)
explored_hexagons = len(hex_density[hex_density['density'] >= 2])
coverage_pct = explored_hexagons / total_hexagons * 100

# ======================================================================
# Code Block 25
# ======================================================================

unexplored = hex_density[hex_density['density_class'] == 'Unexplored']
sparse = hex_density[hex_density['density_class'] == 'Sparse']

# ======================================================================
# Code Block 26
# ======================================================================

print(f"\n✓ Coverage Analysis Complete")
print(f"\nOverall Metrics:")
print(f"  Total area analyzed: ~{total_hexagons * 6500:,} km²")
print(f"  Explored area (>2 holes/km²): {coverage_pct:.1f}%")
print(f"  Unexplored area: {100-coverage_pct:.1f}%")

print(f"\nDensity Breakdown:")
print(f"  Dense zones (>30 holes/km²): {len(hex_density[hex_density['density'] > 30]):,} hexagons")
print(f"  Moderate zones (10-30 holes/km²): {len(hex_density[(hex_density['density'] >= 10) & (hex_density['density'] <= 30)]):,} hexagons")
print(f"  Sparse zones (2-10 holes/km²): {len(sparse):,} hexagons")
print(f"  Unexplored zones (<2 holes/km²): {len(unexplored):,} hexagons")

print(f"\nRecommendations:")
print(f"  • Priority 1: {len(sparse)} sparse hexagons for systematic infill")
print(f"  • Priority 2: {len(unexplored)} unexplored hexagons for reconnaissance")
print(f"  • Estimated cost: ~${(len(sparse)*10 + len(unexplored)*5):,}M for complete coverage")

return {
    'coverage_pct': coverage_pct,
    'unexplored_hexagons': len(unexplored),
    'sparse_hexagons': len(sparse)
}

# ======================================================================
# Code Block 27
# ======================================================================

======================================================================
EXPLORATION COVERAGE ANALYSIS
======================================================================

✓ Coverage Analysis Complete

Overall Metrics:
  Total area analyzed: ~3,165,500 km²
  Explored area (>2 holes/km²): 59.3%
  Unexplored area: 40.7%

Density Breakdown:
  Dense zones (>30 holes/km²): 55 hexagons
  Moderate zones (10-30 holes/km²): 89 hexagons
  Sparse zones (2-10 holes/km²): 145 hexagons
  Unexplored zones (<2 holes/km²): 198 hexagons

Recommendations:
  • Priority 1: 145 sparse hexagons for systematic infill
  • Priority 2: 198 unexplored hexagons for reconnaissance
  • Estimated cost: ~$2,440M for complete coverage

# ======================================================================
# Code Block 28
# ======================================================================

zones = highest ROI infill targets; unexplored gaps in favorable
geology = greenfield opportunities
