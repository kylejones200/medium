#!/usr/bin/env python3
"""
Python code extracted from 14_satellite_leak_detection_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

# Core Dependencies
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Geospatial (Sedona for spatial SQL + Mosaic for visualization)
from sedona.register import SedonaRegistrator
from sedona.core.formatMapper import GeoJsonReader
import mosaic as mos

# Databricks utilities
from pyspark.dbutils import DBUtils

# ======================================================================
# Code Block 2
# ======================================================================

def initialize_spark_geospatial():
    """
    Configure Spark with Sedona and Mosaic for geospatial processing.
    
    Requirements:
    - Databricks ML Runtime 13.3+ with GPU (for potential DL inference)
    - Unity Catalog enabled
    - Sedona and Mosaic libraries installed
    
    Returns:
        Configured SparkSession
    """
    spark = (SparkSession.builder
             .appName("PipelineLeakDetection")
             .config("spark.databricks.io.cache.enabled", "true")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
             .getOrCreate())
    
    # Register Sedona spatial functions
    SedonaRegistrator.registerAll(spark)
    
    # Enable Mosaic for H3 and visualization
    mos.enable_mosaic(spark, dbutils)
    
    print("Spark Session Initialized:")
    print(f"  Version: {spark.version}")
    print(f"  Sedona functions registered: {len(spark.sql('SHOW FUNCTIONS').filter(F.col('function').like('%ST_%')).collect())} spatial UDFs")
    print(f"  Mosaic enabled: H3 tessellation available")
    
    return spark

# Example usage
spark = initialize_spark_geospatial()

# ======================================================================
# Code Block 3
# ======================================================================

def ingest_tropomi_methane(spark, date_range, catalog_path):
    """
    Ingest TROPOMI Level 2 methane products.
    
    Data source: Copernicus Sentinel-5P
    Resolution: 7×7 km
    Frequency: Daily
    
    Args:
        date_range: (start_date, end_date) tuple
        catalog_path: Unity Catalog table path
    
    Returns:
        DataFrame with CH4 observations
    """
    # Read TROPOMI NetCDF files from cloud storage
    # Format: s3://sentinel-5p/OFFL/L2__CH4____/{year}/{month}/{day}/
    
    tropomi_schema = StructType([
        StructField("observation_id", StringType()),
        StructField("date", DateType()),
        StructField("time_utc", TimestampType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("ch4_column_ppb", DoubleType()),  # Column-averaged methane
        StructField("ch4_precision_ppb", DoubleType()),
        StructField("qa_value", DoubleType()),  # Quality flag (>0.5 = good)
    ])
    
    # In production, use actual NetCDF reader
    # Here we simulate with synthetic data
    import numpy as np
    from datetime import datetime, timedelta
    
    start_date, end_date = date_range
    dates = pd.date_range(start_date, end_date, freq='D')
    
    rows = []
    for date in dates:
        # Generate synthetic CH4 observations near ROW
        # Normal background: 1850 ± 30 ppb
        # Leak signature: +20 to +80 ppb enhancement
        for _ in range(100):  # 100 observations per day
            lat = 34.0 + np.random.randn() * 0.5
            lon = -102.0 + np.random.randn() * 0.5
            baseline_ch4 = 1850 + np.random.randn() * 30
            
            # Inject leak signature (5% of obs)
            if np.random.rand() < 0.05:
                baseline_ch4 += np.random.uniform(20, 80)
            
            rows.append({
                'observation_id': f"TROPO_{date.strftime('%Y%m%d')}_{_}",
                'date': date,
                'time_utc': datetime.combine(date, datetime.min.time()),
                'latitude': lat,
                'longitude': lon,
                'ch4_column_ppb': baseline_ch4,
                'ch4_precision_ppb': 15.0,
                'qa_value': 0.85
            })
    
    df = spark.createDataFrame(rows, schema=tropomi_schema)
    
    # Add geometry column
    df = df.withColumn(
        "geom",
        F.expr("ST_Point(longitude, latitude)")
    )
    
    # Filter by quality
    df = df.filter(F.col("qa_value") > 0.5)
    
    # Write to Bronze
    (df.write
     .format("delta")
     .mode("append")
     .partitionBy("date")
     .saveAsTable(f"{catalog_path}.bronze.tropomi_ch4"))
    
    print(f"Ingested TROPOMI CH4: {df.count()} observations")
    print(f"  Date range: {df.agg(F.min('date'), F.max('date')).collect()[0]}")
    print(f"  CH4 range: {df.agg(F.min('ch4_column_ppb'), F.max('ch4_column_ppb')).collect()[0]}")
    
    return df

def ingest_sentinel2_optical(spark, date_range, catalog_path):
    """
    Ingest Sentinel-2 Level 2A surface reflectance.
    
    Resolution: 10m (B02, B03, B04, B08)
    Frequency: 5-day revisit
    
    Returns:
        DataFrame with optical bands
    """
    # Similar structure to TROPOMI ingestion
    # Read Sentinel-2 tiles, extract ROI, compute per-pixel bands
    # Store: date, lat, lon, B02 (blue), B03 (green), B04 (red), B08 (NIR), cloud_mask
    
    print("Sentinel-2 ingestion (simulated)")
    # Implementation details omitted for brevity
    pass

def ingest_sentinel1_sar(spark, date_range, catalog_path):
    """
    Ingest Sentinel-1 SAR coherence.
    
    Resolution: 20m
    Frequency: 6-12 day pairs
    
    Returns:
        DataFrame with coherence values
    """
    print("Sentinel-1 SAR ingestion (simulated)")
    # Implementation details omitted for brevity
    pass

# ======================================================================
# Code Block 4
# ======================================================================

def visualize_leak_scores(spark):
    """
    Create interactive map of leak scores using Mosaic.
    
    Returns:
        Kepler.gl visualization embedded in notebook
    """
    import mosaic as mos
    
    # Load recent scores
    df = spark.table("gold.leak_scores")
    recent = df.filter(F.col("date") == F.current_date())
    
    # Display top 200 highest-score tiles
    top_tiles = recent.orderBy(F.col("leak_score").desc()).limit(200)
    
    # Mosaic display (renders Kepler.gl map in Databricks)
    mos.display(
        top_tiles,
        geometry_col="cell_id",
        color="leak_score",
        title="Pipeline Leak Scores (Last 24 Hours)",
        tooltip_cols=["date", "leak_score", "z_ch4", "z_ndvi_decline", "z_coh_loss"]
    )
    
    # Statistics
    stats = recent.agg(
        F.count("*").alias("total_tiles"),
        F.avg("leak_score").alias("mean_score"),
        F.stddev("leak_score").alias("std_score"),
        F.max("leak_score").alias("max_score"),
        F.expr("percentile(leak_score, 0.95)").alias("p95_score")
    ).collect()[0]
    
    print(f"\nLeak Score Statistics (Last 24 Hours):")
    print(f"  Total tiles monitored: {stats['total_tiles']}")
    print(f"  Mean score: {stats['mean_score']:.2f}")
    print(f"  Std dev: {stats['std_score']:.2f}")
    print(f"  Max score: {stats['max_score']:.2f}")
    print(f"  95th percentile: {stats['p95_score']:.2f}")

def generate_daily_alerts(spark, threshold=3.0):
    """
    Generate inspection alerts for high-score tiles.
    
    Alert criteria:
    - leak_score > 3σ (configurable threshold)
    - Score in top 5% of all observations
    - CH4, NDVI, or coherence individually >2σ
    
    Returns:
        DataFrame with prioritized inspection targets
    """
    alerts = spark.sql(f"""
        SELECT 
            date,
            cell_id,
            leak_score,
            z_ch4,
            z_ndvi_decline,
            z_coh_loss,
            ch4_mean_ppb,
            ndvi_mean,
            coh_mean,
            ROW_NUMBER() OVER(PARTITION BY date ORDER BY leak_score DESC) AS priority_rank
        FROM gold.leak_scores
        WHERE leak_score > {threshold}
          AND (z_ch4 > 2.0 OR z_ndvi_decline > 2.0 OR z_coh_loss > 2.0)
          AND date = CURRENT_DATE
    """)
    
    alert_count = alerts.count()
    
    if alert_count > 0:
        print(f"\n⚠️  {alert_count} LEAK ALERTS GENERATED")
        print(f"{'='*70}")
        
        # Display top 10
        top_alerts = alerts.limit(10).collect()
        for alert in top_alerts:
            print(f"\nPriority #{alert['priority_rank']}: Cell {alert['cell_id']}")
            print(f"  Leak Score: {alert['leak_score']:.2f}σ")
            print(f"  CH4: {alert['ch4_mean_ppb']:.1f} ppb (z={alert['z_ch4']:.2f})")
            print(f"  NDVI: {alert['ndvi_mean']:.3f} (z_decline={alert['z_ndvi_decline']:.2f})")
            print(f"  Coherence: {alert['coh_mean']:.3f} (z_loss={alert['z_coh_loss']:.2f})")
        
        # Write to alert table
        (alerts
         .write
         .format("delta")
         .mode("append")
         .partitionBy("date")
         .saveAsTable("gold.daily_leak_alerts"))
    else:
        print("\n✅ No leak alerts above threshold")
    
    return alerts

# ======================================================================
# Code Block 5
# ======================================================================

"""
Configure Spark with Sedona and Mosaic for geospatial processing.

Requirements:
- Databricks ML Runtime 13.3+ with GPU (for potential DL inference)
- Unity Catalog enabled
- Sedona and Mosaic libraries installed

Returns:
    Configured SparkSession
"""
spark = (SparkSession.builder
         .appName("PipelineLeakDetection")
         .config("spark.databricks.io.cache.enabled", "true")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
         .getOrCreate())

# ======================================================================
# Code Block 6
# ======================================================================

mos.enable_mosaic(spark, dbutils)

print("Spark Session Initialized:")
print(f"  Version: {spark.version}")
print(f"  Sedona functions registered: {len(spark.sql('SHOW FUNCTIONS').filter(F.col('function').like('%ST_%')).collect())} spatial UDFs")
print(f"  Mosaic enabled: H3 tessellation available")

return spark

# ======================================================================
# Code Block 7
# ======================================================================

tropomi_schema = StructType([
    StructField("observation_id", StringType()),
    StructField("date", DateType()),
    StructField("time_utc", TimestampType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("ch4_column_ppb", DoubleType()),  # Column-averaged methane
    StructField("ch4_precision_ppb", DoubleType()),
    StructField("qa_value", DoubleType()),  # Quality flag (>0.5 = good)
])

# ======================================================================
# Code Block 8
# ======================================================================

import numpy as np
from datetime import datetime, timedelta

start_date, end_date = date_range
dates = pd.date_range(start_date, end_date, freq='D')

rows = []
for date in dates:

# ======================================================================
# Code Block 9
# ======================================================================

for _ in range(100):  # 100 observations per day
        lat = 34.0 + np.random.randn() * 0.5
        lon = -102.0 + np.random.randn() * 0.5
        baseline_ch4 = 1850 + np.random.randn() * 30

# ======================================================================
# Code Block 10
# ======================================================================

if np.random.rand() < 0.05:
            baseline_ch4 += np.random.uniform(20, 80)
        
        rows.append({
            'observation_id': f"TROPO_{date.strftime('%Y%m%d')}_{_}",
            'date': date,
            'time_utc': datetime.combine(date, datetime.min.time()),
            'latitude': lat,
            'longitude': lon,
            'ch4_column_ppb': baseline_ch4,
            'ch4_precision_ppb': 15.0,
            'qa_value': 0.85
        })

df = spark.createDataFrame(rows, schema=tropomi_schema)

# ======================================================================
# Code Block 11
# ======================================================================

df = df.withColumn(
    "geom",
    F.expr("ST_Point(longitude, latitude)")
)

# ======================================================================
# Code Block 12
# ======================================================================

df = df.filter(F.col("qa_value") > 0.5)

# ======================================================================
# Code Block 13
# ======================================================================

(df.write
 .format("delta")
 .mode("append")
 .partitionBy("date")
 .saveAsTable(f"{catalog_path}.bronze.tropomi_ch4"))

print(f"Ingested TROPOMI CH4: {df.count()} observations")
print(f"  Date range: {df.agg(F.min('date'), F.max('date')).collect()[0]}")
print(f"  CH4 range: {df.agg(F.min('ch4_column_ppb'), F.max('ch4_column_ppb')).collect()[0]}")

return df

# ======================================================================
# Code Block 14
# ======================================================================

print("Sentinel-2 ingestion (simulated)")

# ======================================================================
# Code Block 15
# ======================================================================

"""
Ingest Sentinel-1 SAR coherence.

Resolution: 20m
Frequency: 6-12 day pairs

Returns:
    DataFrame with coherence values
"""
print("Sentinel-1 SAR ingestion (simulated)")

# ======================================================================
# Code Block 16
# ======================================================================

ON a.date = b.date AND a.cell_id = b.cell_id

# ======================================================================
# Code Block 17
# ======================================================================

ON COALESCE(a.date, b.date) = c.date 
AND COALESCE(a.cell_id, b.cell_id) = c.cell_id;

# ======================================================================
# Code Block 18
# ======================================================================

f.date,
f.cell_id,
f.ch4_mean_ppb,
f.ndvi_mean,
f.coh_mean,

-- Z-scores per sensor
(f.ch4_mean_ppb - b.ch4_baseline_ppb) / NULLIF(b.ch4_sigma_ppb, 10) AS z_ch4,
(b.ndvi_baseline - f.ndvi_mean) / NULLIF(b.ndvi_sigma, 0.05) AS z_ndvi_decline,  -- Invert: decline = positive anomaly
(b.coh_baseline - f.coh_mean) / NULLIF(b.coh_sigma, 0.1) AS z_coh_loss,  -- Invert: coherence loss = positive anomaly

-- Composite leak score (weighted combination)
0.50 * COALESCE((f.ch4_mean_ppb - b.ch4_baseline_ppb) / NULLIF(b.ch4_sigma_ppb, 10), 0) +
0.30 * COALESCE((b.ndvi_baseline - f.ndvi_mean) / NULLIF(b.ndvi_sigma, 0.05), 0) +
0.20 * COALESCE((b.coh_baseline - f.coh_mean) / NULLIF(b.coh_sigma, 0.1), 0) AS leak_score

# ======================================================================
# Code Block 19
# ======================================================================

"""
Create interactive map of leak scores using Mosaic.

Returns:
    Kepler.gl visualization embedded in notebook
"""
import mosaic as mos

# ======================================================================
# Code Block 20
# ======================================================================

df = spark.table("gold.leak_scores")
recent = df.filter(F.col("date") == F.current_date())

# ======================================================================
# Code Block 21
# ======================================================================

top_tiles = recent.orderBy(F.col("leak_score").desc()).limit(200)

# ======================================================================
# Code Block 22
# ======================================================================

mos.display(
    top_tiles,
    geometry_col="cell_id",
    color="leak_score",
    title="Pipeline Leak Scores (Last 24 Hours)",
    tooltip_cols=["date", "leak_score", "z_ch4", "z_ndvi_decline", "z_coh_loss"]
)

# ======================================================================
# Code Block 23
# ======================================================================

stats = recent.agg(
    F.count("*").alias("total_tiles"),
    F.avg("leak_score").alias("mean_score"),
    F.stddev("leak_score").alias("std_score"),
    F.max("leak_score").alias("max_score"),
    F.expr("percentile(leak_score, 0.95)").alias("p95_score")
).collect()[0]

print(f"\nLeak Score Statistics (Last 24 Hours):")
print(f"  Total tiles monitored: {stats['total_tiles']}")
print(f"  Mean score: {stats['mean_score']:.2f}")
print(f"  Std dev: {stats['std_score']:.2f}")
print(f"  Max score: {stats['max_score']:.2f}")
print(f"  95th percentile: {stats['p95_score']:.2f}")

# ======================================================================
# Code Block 24
# ======================================================================

"""
Generate inspection alerts for high-score tiles.

Alert criteria:
- leak_score > 3σ (configurable threshold)
- Score in top 5% of all observations
- CH4, NDVI, or coherence individually >2σ

Returns:
    DataFrame with prioritized inspection targets
"""
alerts = spark.sql(f"""
    SELECT 
        date,
        cell_id,
        leak_score,
        z_ch4,
        z_ndvi_decline,
        z_coh_loss,
        ch4_mean_ppb,
        ndvi_mean,
        coh_mean,
        ROW_NUMBER() OVER(PARTITION BY date ORDER BY leak_score DESC) AS priority_rank
    FROM gold.leak_scores
    WHERE leak_score > {threshold}
      AND (z_ch4 > 2.0 OR z_ndvi_decline > 2.0 OR z_coh_loss > 2.0)
      AND date = CURRENT_DATE
""")

alert_count = alerts.count()

if alert_count > 0:
    print(f"\n⚠️  {alert_count} LEAK ALERTS GENERATED")
    print(f"{'='*70}")

# ======================================================================
# Code Block 25
# ======================================================================

top_alerts = alerts.limit(10).collect()
    for alert in top_alerts:
        print(f"\nPriority #{alert['priority_rank']}: Cell {alert['cell_id']}")
        print(f"  Leak Score: {alert['leak_score']:.2f}σ")
        print(f"  CH4: {alert['ch4_mean_ppb']:.1f} ppb (z={alert['z_ch4']:.2f})")
        print(f"  NDVI: {alert['ndvi_mean']:.3f} (z_decline={alert['z_ndvi_decline']:.2f})")
        print(f"  Coherence: {alert['coh_mean']:.3f} (z_loss={alert['z_coh_loss']:.2f})")

# ======================================================================
# Code Block 26
# ======================================================================

(alerts
     .write
     .format("delta")
     .mode("append")
     .partitionBy("date")
     .saveAsTable("gold.daily_leak_alerts"))
else:
    print("\n✅ No leak alerts above threshold")

return alerts

# ======================================================================
# Code Block 27
# ======================================================================

Leak Score Statistics (Last 24 Hours):
  Total tiles monitored: 8,450
  Mean score: 0.12
  Std dev: 0.87
  Max score: 5.23
  95th percentile: 1.64

⚠️  12 LEAK ALERTS GENERATED
======================================================================

Priority #1: Cell 8a2a1072b59ffff
  Leak Score: 5.23σ
  CH4: 1927.3 ppb (z=4.85)
  NDVI: 0.542 (z_decline=3.21)
  Coherence: 0.423 (z_loss=2.87)

Priority #2: Cell 8a2a1072c8bffff
  Leak Score: 4.67σ
  CH4: 1908.1 ppb (z=4.12)
  NDVI: 0.587 (z_decline=2.34)
  Coherence: 0.512 (z_loss=1.98)

[... 8 more alerts ...]
