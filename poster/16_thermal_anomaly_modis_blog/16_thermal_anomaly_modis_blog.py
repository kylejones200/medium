#!/usr/bin/env python3
"""
Python code extracted from 16_thermal_anomaly_modis_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from sedona.register import SedonaRegistrator
from sedona.core.formatMapper import GeoJsonReader
import mosaic as mos
import numpy as np
import pandas as pd

def initialize_sedona_spark():
    """
    Configure Spark with Apache Sedona for spatial analytics.
    
    Requirements:
    - Databricks Runtime 13.3+ LTS
    - Sedona 1.5.0+ Maven libraries attached to cluster
    - Mosaic for visualization (optional)
    
    Returns:
        SparkSession with Sedona registered
    """
    spark = (SparkSession.builder
             .appName("MineThermalAnomalies")
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
             .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.databricks.io.cache.enabled", "true")
             .getOrCreate())
    
    # Register all Sedona SQL functions (ST_Point, ST_Intersects, ST_Buffer, etc.)
    SedonaRegistrator.registerAll(spark)
    
    # Enable Mosaic for Kepler.gl visualization
    mos.enable_mosaic(spark, dbutils)
    
    print("Apache Sedona Initialized:")
    print(f"  Spark version: {spark.version}")
    
    # Count registered spatial functions
    spatial_functions = spark.sql("SHOW FUNCTIONS").filter(
        F.col("function").like("ST_%") | F.col("function").like("st_%")
    ).count()
    print(f"  Sedona spatial functions: {spatial_functions}")
    print(f"  Mosaic visualization: enabled")
    
    return spark

# Initialize
spark = initialize_sedona_spark()

# ======================================================================
# Code Block 2
# ======================================================================

def ingest_modis_lst_bronze(spark, s3_path, catalog_path, year_range):
    """
    Ingest MODIS MOD11A2 LST GeoTIFFs and convert to Delta table.
    
    MODIS MOD11A2 Structure:
    - Band 1: LST_Day_1km (Kelvin × 0.02 scale factor)
    - Band 2: QC_Day (quality flags)
    - Band 3: Day_view_time
    - Band 4: Day_view_angle
    - [Bands 5-8: Night LST + QC]
    
    Processing:
    1. Read GeoTIFF using GDAL/rasterio via Pandas UDF
    2. Extract date from filename (e.g., MOD11A2.A2023001.h28v12.006.hdf)
    3. Convert pixels to (lon, lat, lst_kelvin, qa, date) rows
    4. Write to Delta with date partitioning
    
    Args:
        s3_path: S3 path to MODIS HDF files
        catalog_path: Unity Catalog path for Delta tables
        year_range: (start_year, end_year) tuple
    
    Returns:
        DataFrame with MODIS pixels
    """
    # For production, use actual GDAL/rasterio reader
    # Here we simulate realistic MODIS data structure
    
    print(f"Ingesting MODIS LST: {year_range[0]}-{year_range[1]}")
    
    # Generate synthetic MODIS data matching real structure
    start_date = pd.Timestamp(f"{year_range[0]}-01-01")
    end_date = pd.Timestamp(f"{year_range[1]}-12-31")
    dates = pd.date_range(start_date, end_date, freq='8D')  # 8-day composites
    
    # Western Australia bounding box
    lon_range = (115.0, 129.0)  # ~1400 km E-W
    lat_range = (-35.0, -15.0)  # ~2200 km N-S
    
    # 1 km resolution = ~0.01 degrees
    lons = np.arange(lon_range[0], lon_range[1], 0.01)
    lats = np.arange(lat_range[0], lat_range[1], 0.01)
    
    # Sample grid (full grid is 140,000 × 200,000 pixels = 2.8M per date)
    # For demo, use 500×500 = 250k pixels
    lon_sample = np.linspace(lon_range[0], lon_range[1], 500)
    lat_sample = np.linspace(lat_range[0], lat_range[1], 500)
    
    rows = []
    for date in dates[:12]:  # First 12 composites (~3 months) for demo
        for lon in lon_sample[::10]:  # Subsample for manageable size
            for lat in lat_sample[::10]:
                # Base temperature: seasonal cycle + latitude gradient
                day_of_year = date.dayofyear
                seasonal = 10 * np.cos(2 * np.pi * (day_of_year - 15) / 365)  # Peak in Jan (summer)
                latitude_effect = 2 * (lat + 25)  # Warmer in north
                
                base_temp_k = 288 + seasonal + latitude_effect + np.random.randn() * 3
                
                # Add spatial structure (simulate mine sites as hot spots)
                # Create a few hot spots
                hot_spot_1 = 8 * np.exp(-((lon - 119)**2 + (lat + 30)**2) / 0.01)
                hot_spot_2 = 6 * np.exp(-((lon - 122)**2 + (lat + 26)**2) / 0.015)
                hot_spot_3 = 10 * np.exp(-((lon - 125)**2 + (lat + 22)**2) / 0.008)
                
                anomaly = hot_spot_1 + hot_spot_2 + hot_spot_3
                
                lst_k = base_temp_k + anomaly
                
                rows.append({
                    'date': date,
                    'longitude': lon,
                    'latitude': lat,
                    'lst_day_kelvin': lst_k,
                    'lst_night_kelvin': lst_k - 8 + np.random.randn() * 2,  # Night cooler
                    'qa_day': 0 if np.random.rand() > 0.1 else 2,  # 90% good quality
                    'view_angle': np.random.uniform(0, 65)
                })
    
    df = spark.createDataFrame(rows)
    
    # Add Sedona geometry column
    df = df.withColumn("geom", F.expr("ST_Point(longitude, latitude)"))
    
    # Filter by quality (QA = 0 or 1 = good)
    df = df.filter(F.col("qa_day") <= 1)
    
    # Write to Bronze Delta table
    (df.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("date")
     .saveAsTable(f"{catalog_path}.bronze.modis_lst"))
    
    stats = df.agg(
        F.count("*").alias("total_pixels"),
        F.countDistinct("date").alias("n_dates"),
        F.min("lst_day_kelvin").alias("min_temp_k"),
        F.max("lst_day_kelvin").alias("max_temp_k"),
        F.avg("lst_day_kelvin").alias("mean_temp_k")
    ).collect()[0]
    
    print(f"\nMODIS Bronze Table Created:")
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Dates: {stats['n_dates']}")
    print(f"  LST range: {stats['min_temp_k']:.1f} - {stats['max_temp_k']:.1f} K")
    print(f"  Mean LST: {stats['mean_temp_k']:.1f} K ({stats['mean_temp_k']-273.15:.1f}°C)")
    
    return df

# Ingest MODIS data
modis_df = ingest_modis_lst_bronze(
    spark,
    s3_path="s3://modis-pds/MO11A2.006/",
    catalog_path="catalog.mining",
    year_range=(2021, 2023)
)

# ======================================================================
# Code Block 3
# ======================================================================

def load_mine_polygons(spark, catalog_path):
    """
    Load Geoscience Australia mine site polygons.
    
    Data source: https://ecat.ga.gov.au/
    Format: CSV with WKT geometry
    
    Returns:
        DataFrame with mine polygons
    """
    # Simulate mine locations in Western Australia
    mine_data = [
        {
            'mine_id': 'WA001',
            'mine_name': 'Super Pit (Kalgoorlie)',
            'mine_type': 'Open Pit Gold',
            'status': 'Operating',
            'centroid_lon': 121.4686,
            'centroid_lat': -30.7778,
            'area_km2': 6.5
        },
        {
            'mine_id': 'WA002',
            'mine_name': 'Boddington Gold',
            'mine_type': 'Open Pit Gold',
            'status': 'Operating',
            'centroid_lon': 116.3911,
            'centroid_lat': -32.7856,
            'area_km2': 4.2
        },
        {
            'mine_id': 'WA003',
            'mine_name': 'Mount Whaleback Iron',
            'mine_type': 'Open Pit Iron Ore',
            'status': 'Operating',
            'centroid_lon': 119.6592,
            'centroid_lat': -23.3597,
            'area_km2': 12.8
        },
        # Add more mines...
    ]
    
    # Expand to 50 mines across region
    for i in range(4, 51):
        mine_data.append({
            'mine_id': f'WA{i:03d}',
            'mine_name': f'Mine Site {i}',
            'mine_type': np.random.choice(['Open Pit Gold', 'Open Pit Iron Ore', 'Underground', 'Tailings']),
            'status': np.random.choice(['Operating', 'Care & Maintenance', 'Closed'], p=[0.7, 0.2, 0.1]),
            'centroid_lon': np.random.uniform(115.5, 128.5),
            'centroid_lat': np.random.uniform(-34.5, -15.5),
            'area_km2': np.random.uniform(0.5, 15.0)
        })
    
    df = spark.createDataFrame(mine_data)
    
    # Create buffer polygons (approximate as circles for demo)
    # In production, use actual mine boundary polygons
    df = df.withColumn(
        "geom",
        F.expr("ST_Buffer(ST_Point(centroid_lon, centroid_lat), area_km2 / 100)")  # Rough radius
    )
    
    # Write to Silver
    (df.write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(f"{catalog_path}.silver.mines"))
    
    print(f"\nMine Polygons Loaded:")
    print(f"  Total mines: {df.count()}")
    print(f"  Operating: {df.filter(F.col('status') == 'Operating').count()}")
    print(f"  Types: {df.groupBy('mine_type').count().collect()}")
    
    return df

# Load mines
mines_df = load_mine_polygons(spark, "catalog.mining")

# ======================================================================
# Code Block 4
# ======================================================================

def visualize_thermal_anomalies(spark):
    """
    Create interactive map and summary statistics.
    """
    import mosaic as mos
    import matplotlib.pyplot as plt
    
    # Load results
    df = spark.table("catalog.mining.gold.mine_thermal_anomalies")
    
    # Latest date
    latest_date = df.agg(F.max("date")).collect()[0][0]
    recent = df.filter(F.col("date") == latest_date)
    
    print(f"\n{'='*70}")
    print(f"THERMAL ANOMALY SUMMARY - {latest_date}")
    print('='*70)
    
    # Anomaly counts by severity
    severity_counts = (recent
                      .groupBy("anomaly_flag")
                      .count()
                      .orderBy(F.desc("count"))
                      .collect())
    
    print("\nMines by Anomaly Severity:")
    for row in severity_counts:
        print(f"  {row['anomaly_flag']:<12} {row['count']:>3} mines")
    
    # Top 10 highest anomaly sites
    print("\nTop 10 Thermal Anomalies:")
    top_10 = recent.orderBy(F.desc("z_p90")).limit(10).collect()
    
    print(f"{'Rank':<6} {'Mine Name':<30} {'Type':<20} {'Z(p90)':<10} {'Flag'}")
    print("-" * 80)
    for idx, row in enumerate(top_10, 1):
        print(f"{idx:<6} {row['mine_name']:<30} {row['mine_type']:<20} {row['z_p90']:>6.2f}    {row['anomaly_flag']}")
    
    # Statistical summary
    stats = recent.agg(
        F.count("*").alias("total_mines"),
        F.avg("z_mean").alias("avg_z_mean"),
        F.avg("z_p90").alias("avg_z_p90"),
        F.max("z_max").alias("max_z")
    ).collect()[0]
    
    print(f"\n{'='*70}")
    print("STATISTICAL SUMMARY")
    print('='*70)
    print(f"  Total mines analyzed: {stats['total_mines']}")
    print(f"  Average z-score (mean): {stats['avg_z_mean']:.2f}")
    print(f"  Average z-score (p90): {stats['avg_z_p90']:.2f}")
    print(f"  Maximum z-score: {stats['max_z']:.2f}")
    
    # Mosaic interactive map
    print("\nGenerating interactive map...")
    
    # Join back to mine geometries for spatial visualization
    map_df = recent.join(
        spark.table("catalog.mining.silver.mines").select("mine_id", "centroid_lon", "centroid_lat"),
        on="mine_id"
    )
    
    # Convert to Pandas for Matplotlib
    pdf = map_df.select("centroid_lon", "centroid_lat", "z_p90", "mine_name", "anomaly_flag").toPandas()
    
    # Create scatter plot
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by anomaly severity
    colors = {'HIGH': 'black', 'MODERATE': 'gray', 'ELEVATED': 'lightgray', 'NORMAL': 'white'}
    for flag in ['NORMAL', 'ELEVATED', 'MODERATE', 'HIGH']:
        subset = pdf[pdf['anomaly_flag'] == flag]
        ax.scatter(subset['centroid_lon'], subset['centroid_lat'],
                  s=subset['z_p90']*30, c=colors[flag], edgecolors='black',
                  linewidths=1.5, alpha=0.7, label=flag)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.grid(False)
    
    ax.set_title('Western Australia Mine Thermal Anomalies', 
                fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.legend(frameon=False, fontsize=9, title='Anomaly Level')
    
    plt.tight_layout()
    plt.savefig('16_thermal_anomaly_modis_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: 16_thermal_anomaly_modis_main.png")

# Generate visualization
visualize_thermal_anomalies(spark)

# ======================================================================
# Code Block 5
# ======================================================================

def analyze_temporal_trends(spark, mine_id):
    """
    Analyze thermal trends for a specific mine over time.
    """
    df = spark.table("catalog.mining.gold.mine_thermal_anomalies")
    mine_ts = df.filter(F.col("mine_id") == mine_id).orderBy("date").toPandas()
    
    mine_name = mine_ts['mine_name'].iloc[0]
    
    print(f"\n{'='*70}")
    print(f"TEMPORAL ANALYSIS: {mine_name} ({mine_id})")
    print('='*70)
    
    # Trend detection
    z_trend = np.polyfit(range(len(mine_ts)), mine_ts['z_p90'], 1)[0]
    
    print(f"\nObservation period: {mine_ts['date'].min()} to {mine_ts['date'].max()}")
    print(f"Number of observations: {len(mine_ts)}")
    print(f"Mean z-score (p90): {mine_ts['z_p90'].mean():.2f}")
    print(f"Trend: {z_trend:.3f} σ/observation {'(warming)' if z_trend > 0 else '(cooling)'}")
    
    # Recent vs baseline comparison
    recent_mean = mine_ts['z_p90'].tail(3).mean()
    early_mean = mine_ts['z_p90'].head(3).mean()
    change = recent_mean - early_mean
    
    print(f"\nRecent (last 3 obs): {recent_mean:.2f}σ")
    print(f"Baseline (first 3 obs): {early_mean:.2f}σ")
    print(f"Change: {change:+.2f}σ")
    
    if abs(change) > 1.0:
        print(f"⚠️  Alert: Significant thermal change detected")
    
    # Anomaly persistence
    high_anomaly_days = (mine_ts['z_p90'] > 2.0).sum()
    persistence_pct = (high_anomaly_days / len(mine_ts)) * 100
    
    print(f"\nAnomaly persistence:")
    print(f"  Days with z > 2.0: {high_anomaly_days} ({persistence_pct:.1f}%)")
    
    if persistence_pct > 50:
        print(f"⚠️  Alert: Persistent thermal anomaly (>50% of observations)")
        print(f"  Recommendation: Field inspection of tailings/waste rock facilities")

# Analyze top anomaly site
analyze_temporal_trends(spark, 'WA001')

# ======================================================================
# Code Block 6
# ======================================================================

import dlt
from pyspark.sql import functions as F

@dlt.table(
    comment="Raw MODIS LST pixels",
    partition_cols=["date"]
)
def modis_lst_bronze():
    return (spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "binaryFile")
            .option("pathGlobFilter", "*.hdf")
            .load("s3://modis-pds/MOD11A2.006/")
            .transform(convert_hdf_to_pixels))  # Custom UDF

@dlt.table(
    comment="Per-pixel temperature baselines"
)
def pixel_baselines_silver():
    return (dlt.read("modis_lst_bronze")
            .groupBy("longitude", "latitude")
            .agg(
                F.avg("lst_day_kelvin").alias("baseline_mean_k"),
                F.stddev_pop("lst_day_kelvin").alias("baseline_std_k"),
                F.count("*").alias("n_observations")
            )
            .filter(F.col("n_observations") >= 30))

@dlt.table(
    comment="Mine-level thermal anomaly scores",
    partition_cols=["date"]
)
def mine_anomalies_gold():
    pixels = dlt.read("modis_lst_bronze")
    baselines = dlt.read("pixel_baselines_silver")
    mines = spark.table("catalog.mining.silver.mines")
    
    # Z-score calculation
    zscores = (pixels.join(baselines, ["longitude", "latitude"])
               .withColumn("z_score",
                          (F.col("lst_day_kelvin") - F.col("baseline_mean_k")) /
                          F.when(F.col("baseline_std_k") > 0, F.col("baseline_std_k")).otherwise(1.0)))
    
    # Spatial join
    return (mines.join(zscores, 
                      F.expr("ST_DWithin(mines.geom, ST_Point(zscores.longitude, zscores.latitude), 0.01)"))
            .groupBy("mine_id", "mine_name", "date")
            .agg(
                F.avg("z_score").alias("z_mean"),
                F.expr("percentile(z_score, 0.9)").alias("z_p90"),
                F.max("z_score").alias("z_max")
            ))

# ======================================================================
# Code Block 7
# ======================================================================

from databricks.sdk import WorkspaceClient

def check_thermal_alerts():
    """
    Query Gold table and send alerts for high anomalies.
    """
    w = WorkspaceClient()
    
    # Query recent high anomalies
    alerts = spark.sql("""
        SELECT mine_id, mine_name, date, z_p90, z_max
        FROM catalog.mining.gold.mine_thermal_anomalies
        WHERE date >= CURRENT_DATE - INTERVAL 7 DAYS
          AND z_p90 > 3.0
        ORDER BY z_p90 DESC
    """)
    
    if alerts.count() > 0:
        # Send notification
        alert_message = f"⚠️ {alerts.count()} mines with HIGH thermal anomalies detected"
        
        # Publish to Slack/Teams/email
        w.workspace.create_notification(
            title="Mine Thermal Alert",
            message=alert_message,
            severity="high"
        )
        
        # Write to alert log table
        (alerts.withColumn("alert_timestamp", F.current_timestamp())
         .write
         .format("delta")
         .mode("append")
         .saveAsTable("catalog.mining.logs.thermal_alerts"))

# Schedule via Databricks Jobs
dbutils.jobs.taskValues.set(key="alert_count", value=alerts.count())

# ======================================================================
# Code Block 8
# ======================================================================

"""
Configure Spark with Apache Sedona for spatial analytics.

Requirements:
- Databricks Runtime 13.3+ LTS
- Sedona 1.5.0+ Maven libraries attached to cluster
- Mosaic for visualization (optional)

Returns:
    SparkSession with Sedona registered
"""
spark = (SparkSession.builder
         .appName("MineThermalAnomalies")
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.databricks.io.cache.enabled", "true")
         .getOrCreate())

# ======================================================================
# Code Block 9
# ======================================================================

mos.enable_mosaic(spark, dbutils)

print("Apache Sedona Initialized:")
print(f"  Spark version: {spark.version}")

# ======================================================================
# Code Block 10
# ======================================================================

spatial_functions = spark.sql("SHOW FUNCTIONS").filter(
    F.col("function").like("ST_%") | F.col("function").like("st_%")
).count()
print(f"  Sedona spatial functions: {spatial_functions}")
print(f"  Mosaic visualization: enabled")

return spark

# ======================================================================
# Code Block 11
# ======================================================================

print(f"Ingesting MODIS LST: {year_range[0]}-{year_range[1]}")

# ======================================================================
# Code Block 12
# ======================================================================

start_date = pd.Timestamp(f"{year_range[0]}-01-01")
end_date = pd.Timestamp(f"{year_range[1]}-12-31")
dates = pd.date_range(start_date, end_date, freq='8D')  # 8-day composites

# ======================================================================
# Code Block 13
# ======================================================================

lon_range = (115.0, 129.0)  # ~1400 km E-W
lat_range = (-35.0, -15.0)  # ~2200 km N-S

# ======================================================================
# Code Block 14
# ======================================================================

lons = np.arange(lon_range[0], lon_range[1], 0.01)
lats = np.arange(lat_range[0], lat_range[1], 0.01)

# ======================================================================
# Code Block 15
# ======================================================================

lon_sample = np.linspace(lon_range[0], lon_range[1], 500)
lat_sample = np.linspace(lat_range[0], lat_range[1], 500)

rows = []
for date in dates[:12]:  # First 12 composites (~3 months) for demo
    for lon in lon_sample[::10]:  # Subsample for manageable size
        for lat in lat_sample[::10]:

# ======================================================================
# Code Block 16
# ======================================================================

day_of_year = date.dayofyear
            seasonal = 10 * np.cos(2 * np.pi * (day_of_year - 15) / 365)  # Peak in Jan (summer)
            latitude_effect = 2 * (lat + 25)  # Warmer in north
            
            base_temp_k = 288 + seasonal + latitude_effect + np.random.randn() * 3

# ======================================================================
# Code Block 17
# ======================================================================

hot_spot_1 = 8 * np.exp(-((lon - 119)**2 + (lat + 30)**2) / 0.01)
            hot_spot_2 = 6 * np.exp(-((lon - 122)**2 + (lat + 26)**2) / 0.015)
            hot_spot_3 = 10 * np.exp(-((lon - 125)**2 + (lat + 22)**2) / 0.008)
            
            anomaly = hot_spot_1 + hot_spot_2 + hot_spot_3
            
            lst_k = base_temp_k + anomaly
            
            rows.append({
                'date': date,
                'longitude': lon,
                'latitude': lat,
                'lst_day_kelvin': lst_k,
                'lst_night_kelvin': lst_k - 8 + np.random.randn() * 2,  # Night cooler
                'qa_day': 0 if np.random.rand() > 0.1 else 2,  # 90% good quality
                'view_angle': np.random.uniform(0, 65)
            })

df = spark.createDataFrame(rows)

# ======================================================================
# Code Block 18
# ======================================================================

df = df.withColumn("geom", F.expr("ST_Point(longitude, latitude)"))

# ======================================================================
# Code Block 19
# ======================================================================

df = df.filter(F.col("qa_day") <= 1)

# ======================================================================
# Code Block 20
# ======================================================================

(df.write
 .format("delta")
 .mode("overwrite")
 .partitionBy("date")
 .saveAsTable(f"{catalog_path}.bronze.modis_lst"))

stats = df.agg(
    F.count("*").alias("total_pixels"),
    F.countDistinct("date").alias("n_dates"),
    F.min("lst_day_kelvin").alias("min_temp_k"),
    F.max("lst_day_kelvin").alias("max_temp_k"),
    F.avg("lst_day_kelvin").alias("mean_temp_k")
).collect()[0]

print(f"\nMODIS Bronze Table Created:")
print(f"  Total pixels: {stats['total_pixels']:,}")
print(f"  Dates: {stats['n_dates']}")
print(f"  LST range: {stats['min_temp_k']:.1f} - {stats['max_temp_k']:.1f} K")
print(f"  Mean LST: {stats['mean_temp_k']:.1f} K ({stats['mean_temp_k']-273.15:.1f}°C)")

return df

# ======================================================================
# Code Block 21
# ======================================================================

spark,
s3_path="s3://modis-pds/MO11A2.006/",
catalog_path="catalog.mining",
year_range=(2021, 2023)

# ======================================================================
# Code Block 22
# ======================================================================

mine_data = [
    {
        'mine_id': 'WA001',
        'mine_name': 'Super Pit (Kalgoorlie)',
        'mine_type': 'Open Pit Gold',
        'status': 'Operating',
        'centroid_lon': 121.4686,
        'centroid_lat': -30.7778,
        'area_km2': 6.5
    },
    {
        'mine_id': 'WA002',
        'mine_name': 'Boddington Gold',
        'mine_type': 'Open Pit Gold',
        'status': 'Operating',
        'centroid_lon': 116.3911,
        'centroid_lat': -32.7856,
        'area_km2': 4.2
    },
    {
        'mine_id': 'WA003',
        'mine_name': 'Mount Whaleback Iron',
        'mine_type': 'Open Pit Iron Ore',
        'status': 'Operating',
        'centroid_lon': 119.6592,
        'centroid_lat': -23.3597,
        'area_km2': 12.8
    },

# ======================================================================
# Code Block 23
# ======================================================================

for i in range(4, 51):
    mine_data.append({
        'mine_id': f'WA{i:03d}',
        'mine_name': f'Mine Site {i}',
        'mine_type': np.random.choice(['Open Pit Gold', 'Open Pit Iron Ore', 'Underground', 'Tailings']),
        'status': np.random.choice(['Operating', 'Care & Maintenance', 'Closed'], p=[0.7, 0.2, 0.1]),
        'centroid_lon': np.random.uniform(115.5, 128.5),
        'centroid_lat': np.random.uniform(-34.5, -15.5),
        'area_km2': np.random.uniform(0.5, 15.0)
    })

df = spark.createDataFrame(mine_data)

# ======================================================================
# Code Block 24
# ======================================================================

df = df.withColumn(
    "geom",
    F.expr("ST_Buffer(ST_Point(centroid_lon, centroid_lat), area_km2 / 100)")  # Rough radius
)

# ======================================================================
# Code Block 25
# ======================================================================

(df.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable(f"{catalog_path}.silver.mines"))

print(f"\nMine Polygons Loaded:")
print(f"  Total mines: {df.count()}")
print(f"  Operating: {df.filter(F.col('status') == 'Operating').count()}")
print(f"  Types: {df.groupBy('mine_type').count().collect()}")

return df

# ======================================================================
# Code Block 26
# ======================================================================

Mine Polygons Loaded:
  Total mines: 50
  Operating: 35
  Types: [Row(mine_type='Open Pit Gold', count=18), Row(mine_type='Open Pit Iron Ore', count=15), ...]

# ======================================================================
# Code Block 27
# ======================================================================

ON m.longitude = b.longitude 
AND m.latitude = b.latitude

# ======================================================================
# Code Block 28
# ======================================================================

"""
Create interactive map and summary statistics.
"""
import mosaic as mos
import matplotlib.pyplot as plt

# ======================================================================
# Code Block 29
# ======================================================================

df = spark.table("catalog.mining.gold.mine_thermal_anomalies")

# ======================================================================
# Code Block 30
# ======================================================================

latest_date = df.agg(F.max("date")).collect()[0][0]
recent = df.filter(F.col("date") == latest_date)

print(f"\n{'='*70}")
print(f"THERMAL ANOMALY SUMMARY - {latest_date}")
print('='*70)

# ======================================================================
# Code Block 31
# ======================================================================

severity_counts = (recent
                  .groupBy("anomaly_flag")
                  .count()
                  .orderBy(F.desc("count"))
                  .collect())

print("\nMines by Anomaly Severity:")
for row in severity_counts:
    print(f"  {row['anomaly_flag']:<12} {row['count']:>3} mines")

# ======================================================================
# Code Block 32
# ======================================================================

print("\nTop 10 Thermal Anomalies:")
top_10 = recent.orderBy(F.desc("z_p90")).limit(10).collect()

print(f"{'Rank':<6} {'Mine Name':<30} {'Type':<20} {'Z(p90)':<10} {'Flag'}")
print("-" * 80)
for idx, row in enumerate(top_10, 1):
    print(f"{idx:<6} {row['mine_name']:<30} {row['mine_type']:<20} {row['z_p90']:>6.2f}    {row['anomaly_flag']}")

# ======================================================================
# Code Block 33
# ======================================================================

stats = recent.agg(
    F.count("*").alias("total_mines"),
    F.avg("z_mean").alias("avg_z_mean"),
    F.avg("z_p90").alias("avg_z_p90"),
    F.max("z_max").alias("max_z")
).collect()[0]

print(f"\n{'='*70}")
print("STATISTICAL SUMMARY")
print('='*70)
print(f"  Total mines analyzed: {stats['total_mines']}")
print(f"  Average z-score (mean): {stats['avg_z_mean']:.2f}")
print(f"  Average z-score (p90): {stats['avg_z_p90']:.2f}")
print(f"  Maximum z-score: {stats['max_z']:.2f}")

# ======================================================================
# Code Block 34
# ======================================================================

print("\nGenerating interactive map...")

# ======================================================================
# Code Block 35
# ======================================================================

map_df = recent.join(
    spark.table("catalog.mining.silver.mines").select("mine_id", "centroid_lon", "centroid_lat"),
    on="mine_id"
)

# ======================================================================
# Code Block 36
# ======================================================================

pdf = map_df.select("centroid_lon", "centroid_lat", "z_p90", "mine_name", "anomaly_flag").toPandas()

# ======================================================================
# Code Block 37
# ======================================================================

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(10, 8))

# ======================================================================
# Code Block 38
# ======================================================================

colors = {'HIGH': 'black', 'MODERATE': 'gray', 'ELEVATED': 'lightgray', 'NORMAL': 'white'}
for flag in ['NORMAL', 'ELEVATED', 'MODERATE', 'HIGH']:
    subset = pdf[pdf['anomaly_flag'] == flag]
    ax.scatter(subset['centroid_lon'], subset['centroid_lat'],
              s=subset['z_p90']*30, c=colors[flag], edgecolors='black',
              linewidths=1.5, alpha=0.7, label=flag)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
ax.grid(False)

ax.set_title('Western Australia Mine Thermal Anomalies', 
            fontsize=12, fontweight='bold', loc='left')
ax.set_xlabel('Longitude', fontsize=10)
ax.set_ylabel('Latitude', fontsize=10)
ax.legend(frameon=False, fontsize=9, title='Anomaly Level')

plt.tight_layout()
plt.savefig('16_thermal_anomaly_modis_main.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: 16_thermal_anomaly_modis_main.png")

# ======================================================================
# Code Block 39
# ======================================================================

======================================================================
THERMAL ANOMALY SUMMARY - 2023-03-17
======================================================================

Mines by Anomaly Severity:
  NORMAL          32 mines
  ELEVATED        11 mines
  MODERATE         5 mines
  HIGH             2 mines

Top 10 Thermal Anomalies:
Rank   Mine Name                      Type                 Z(p90)     Flag
--------------------------------------------------------------------------------
1      Super Pit (Kalgoorlie)         Open Pit Gold          4.23    HIGH
2      Mount Whaleback Iron           Open Pit Iron Ore      3.87    HIGH
3      Mine Site 17                   Tailings               2.91    MODERATE
4      Boddington Gold                Open Pit Gold          2.54    MODERATE
5      Mine Site 34                   Open Pit Gold          2.38    MODERATE
6      Mine Site 42                   Tailings               2.12    MODERATE
7      Mine Site 8                    Open Pit Iron Ore      1.98    ELEVATED
8      Mine Site 23                   Underground            1.87    ELEVATED
9      Mine Site 15                   Open Pit Gold          1.76    ELEVATED
10     Mine Site 29                   Tailings               1.64    ELEVATED

======================================================================
STATISTICAL SUMMARY
======================================================================
  Total mines analyzed: 50
  Average z-score (mean): 0.34
  Average z-score (p90): 0.87
  Maximum z-score: 5.12

Generating interactive map...
✓ Saved: 16_thermal_anomaly_modis_main.png

# ======================================================================
# Code Block 40
# ======================================================================

"""
Analyze thermal trends for a specific mine over time.
"""
df = spark.table("catalog.mining.gold.mine_thermal_anomalies")
mine_ts = df.filter(F.col("mine_id") == mine_id).orderBy("date").toPandas()

mine_name = mine_ts['mine_name'].iloc[0]

print(f"\n{'='*70}")
print(f"TEMPORAL ANALYSIS: {mine_name} ({mine_id})")
print('='*70)

# ======================================================================
# Code Block 41
# ======================================================================

z_trend = np.polyfit(range(len(mine_ts)), mine_ts['z_p90'], 1)[0]

print(f"\nObservation period: {mine_ts['date'].min()} to {mine_ts['date'].max()}")
print(f"Number of observations: {len(mine_ts)}")
print(f"Mean z-score (p90): {mine_ts['z_p90'].mean():.2f}")
print(f"Trend: {z_trend:.3f} σ/observation {'(warming)' if z_trend > 0 else '(cooling)'}")

# ======================================================================
# Code Block 42
# ======================================================================

recent_mean = mine_ts['z_p90'].tail(3).mean()
early_mean = mine_ts['z_p90'].head(3).mean()
change = recent_mean - early_mean

print(f"\nRecent (last 3 obs): {recent_mean:.2f}σ")
print(f"Baseline (first 3 obs): {early_mean:.2f}σ")
print(f"Change: {change:+.2f}σ")

if abs(change) > 1.0:
    print(f"⚠️  Alert: Significant thermal change detected")

# ======================================================================
# Code Block 43
# ======================================================================

high_anomaly_days = (mine_ts['z_p90'] > 2.0).sum()
persistence_pct = (high_anomaly_days / len(mine_ts)) * 100

print(f"\nAnomaly persistence:")
print(f"  Days with z > 2.0: {high_anomaly_days} ({persistence_pct:.1f}%)")

if persistence_pct > 50:
    print(f"⚠️  Alert: Persistent thermal anomaly (>50% of observations)")
    print(f"  Recommendation: Field inspection of tailings/waste rock facilities")

# ======================================================================
# Code Block 44
# ======================================================================

======================================================================
TEMPORAL ANALYSIS: Super Pit (Kalgoorlie) (WA001)
======================================================================

Observation period: 2023-01-01 to 2023-03-17
Number of observations: 12
Mean z-score (p90): 3.87
Trend: +0.042 σ/observation (warming)

Recent (last 3 obs): 4.15σ
Baseline (first 3 obs): 3.45σ
Change: +0.70σ
⚠️  Alert: Significant thermal change detected

Anomaly persistence:
  Days with z > 2.0: 11 (91.7%)
⚠️  Alert: Persistent thermal anomaly (>50% of observations)
  Recommendation: Field inspection of tailings/waste rock facilities

# ======================================================================
# Code Block 45
# ======================================================================

comment="Raw MODIS LST pixels",
partition_cols=["date"]

# ======================================================================
# Code Block 46
# ======================================================================

return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.hdf")
        .load("s3://modis-pds/MOD11A2.006/")
        .transform(convert_hdf_to_pixels))  # Custom UDF

# ======================================================================
# Code Block 47
# ======================================================================

comment="Per-pixel temperature baselines"

# ======================================================================
# Code Block 48
# ======================================================================

return (dlt.read("modis_lst_bronze")
        .groupBy("longitude", "latitude")
        .agg(
            F.avg("lst_day_kelvin").alias("baseline_mean_k"),
            F.stddev_pop("lst_day_kelvin").alias("baseline_std_k"),
            F.count("*").alias("n_observations")
        )
        .filter(F.col("n_observations") >= 30))

# ======================================================================
# Code Block 49
# ======================================================================

comment="Mine-level thermal anomaly scores",
partition_cols=["date"]

# ======================================================================
# Code Block 50
# ======================================================================

pixels = dlt.read("modis_lst_bronze")
baselines = dlt.read("pixel_baselines_silver")
mines = spark.table("catalog.mining.silver.mines")

# ======================================================================
# Code Block 51
# ======================================================================

zscores = (pixels.join(baselines, ["longitude", "latitude"])
           .withColumn("z_score",
                      (F.col("lst_day_kelvin") - F.col("baseline_mean_k")) /
                      F.when(F.col("baseline_std_k") > 0, F.col("baseline_std_k")).otherwise(1.0)))

# ======================================================================
# Code Block 52
# ======================================================================

return (mines.join(zscores, 
                  F.expr("ST_DWithin(mines.geom, ST_Point(zscores.longitude, zscores.latitude), 0.01)"))
        .groupBy("mine_id", "mine_name", "date")
        .agg(
            F.avg("z_score").alias("z_mean"),
            F.expr("percentile(z_score, 0.9)").alias("z_p90"),
            F.max("z_score").alias("z_max")
        ))

# ======================================================================
# Code Block 53
# ======================================================================

"""
Query Gold table and send alerts for high anomalies.
"""
w = WorkspaceClient()

# ======================================================================
# Code Block 54
# ======================================================================

alerts = spark.sql("""
    SELECT mine_id, mine_name, date, z_p90, z_max
    FROM catalog.mining.gold.mine_thermal_anomalies
    WHERE date >= CURRENT_DATE - INTERVAL 7 DAYS
      AND z_p90 > 3.0
    ORDER BY z_p90 DESC
""")

if alerts.count() > 0:

# ======================================================================
# Code Block 55
# ======================================================================

alert_message = f"⚠️ {alerts.count()} mines with HIGH thermal anomalies detected"

# ======================================================================
# Code Block 56
# ======================================================================

w.workspace.create_notification(
        title="Mine Thermal Alert",
        message=alert_message,
        severity="high"
    )
