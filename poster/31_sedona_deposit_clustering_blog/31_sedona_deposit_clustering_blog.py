#!/usr/bin/env python3
"""
Python code extracted from 31_sedona_deposit_clustering_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

# %pip install apache-sedona[spark]==1.5.1

# Import libraries
from sedona.spark import *
from sedona.register import SedonaRegistrator
from sedona.utils.adapter import Adapter
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize Sedona
spark = (SparkSession.builder
    .appName("SedonaDepositClustering")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
    .getOrCreate())

SedonaRegistrator.registerAll(spark)

# print(" Sedona registered")

# ======================================================================
# Code Block 2
# ======================================================================

import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 10000

# Western Australia bounding box
lon_min, lon_max = 115.0, 129.0
lat_min, lat_max = -35.0, -16.0

# Generate samples
lons = np.random.uniform(lon_min, lon_max, n_samples)
lats = np.random.uniform(lat_min, lat_max, n_samples)

# Simulate 3 deposit clusters
cluster_centers = [(120.5, -30.5), (124.2, -22.8), (117.3, -28.1)]
cluster_labels = []

for lon, lat in zip(lons, lats):
    dists = [np.sqrt((lon - cx)**2 + (lat - cy)**2) for cx, cy in cluster_centers]
    nearest_idx = np.argmin(dists)
    nearest_dist = dists[nearest_idx]
    
    # Assign cluster if within 2 degrees
    if nearest_dist < 2.0:
        cluster_labels.append(nearest_idx)
    else:
        cluster_labels.append(-1)  # Background

cluster_labels = np.array(cluster_labels)

# Generate grades based on cluster membership
def generate_grade(cluster_id):
    if cluster_id == -1:  # Background
        return max(0.001, np.random.lognormal(-3, 1))
    elif cluster_id == 0:  # Gold cluster
        return max(0.01, np.random.lognormal(0, 0.8))
    elif cluster_id == 1:  # Copper-gold cluster
        return max(0.01, np.random.lognormal(-0.5, 0.6))
    else:  # Zinc-lead cluster
        return max(0.01, np.random.lognormal(-1, 0.5))

Au_ppm = np.array([generate_grade(c) for c in cluster_labels])
Cu_ppm = np.where(cluster_labels == 1, Au_ppm * 3000, np.random.lognormal(1, 1, n_samples))
Zn_ppm = np.where(cluster_labels == 2, Au_ppm * 5000, np.random.lognormal(2, 1, n_samples))

# Create DataFrame
geochem_df = pd.DataFrame({
    'sample_id': [f'GS{i:06d}' for i in range(n_samples)],
    'longitude': lons,
    'latitude': lats,
    'Au_ppm': Au_ppm,
    'Cu_ppm': Cu_ppm,
    'Zn_ppm': Zn_ppm,
    'sample_type': np.random.choice(['soil', 'stream_sed', 'rock'], n_samples, p=[0.7, 0.2, 0.1])
})

# Convert to Spark DataFrame
geochem = spark.createDataFrame(geochem_df)
geochem.write.format("delta").mode("overwrite").saveAsTable("bronze.geochemical_samples")

# print(f" Loaded {geochem.count():,} samples to bronze.geochemical_samples")

# ======================================================================
# Code Block 3
# ======================================================================

# Read bronze data
samples = spark.table("bronze.geochemical_samples")

# Create spatial geometry column
samples = samples.selectExpr(
    "*",
    "ST_Point(CAST(longitude AS DOUBLE), CAST(latitude AS DOUBLE)) AS geometry"
)

# Filter anomalous samples (multi-element threshold)
anomalous = samples.filter(
    (col("Au_ppm") > 0.1) |
    (col("Cu_ppm") > 100) |
    (col("Zn_ppm") > 200)
)

anomalous.write.format("delta").mode("overwrite").saveAsTable("silver.anomalous_samples")

# print(f" Filtered to {anomalous.count():,} anomalous samples")

# ======================================================================
# Code Block 4
# ======================================================================

# Read anomalous samples
samples_spatial = spark.table("silver.anomalous_samples")
samples_spatial.createOrReplaceTempView("samples")

# Perform DBSCAN clustering
# epsilon = 0.1 degrees (~11 km at equator)
# minPoints = 10 samples

clustered = spark.sql("""
    SELECT 
        sample_id,
        longitude,
        latitude,
        Au_ppm,
        Cu_ppm,
        Zn_ppm,
        geometry,
        ST_ClusterDBSCAN(geometry, 0.1, 10) OVER() AS cluster_id
    FROM samples
""")

clustered.write.format("delta").mode("overwrite").saveAsTable("gold.clustered_samples")

# print(" DBSCAN clustering complete")

# Cluster statistics
cluster_stats = spark.sql("""
    SELECT 
        cluster_id,
        COUNT(*) AS n_samples,
        AVG(Au_ppm) AS avg_Au,
        AVG(Cu_ppm) AS avg_Cu,
        AVG(Zn_ppm) AS avg_Zn,
        MAX(Au_ppm) AS max_Au,
        ST_ConvexHull(ST_Collect(geometry)) AS cluster_polygon
    FROM gold.clustered_samples
    WHERE cluster_id IS NOT NULL
    GROUP BY cluster_id
    ORDER BY n_samples DESC
""")

cluster_stats.show(10, truncate=False)

# ======================================================================
# Code Block 5
# ======================================================================

# Generate synthetic known deposits
deposits_df = pd.DataFrame({
    'deposit_id': ['D001', 'D002', 'D003'],
    'name': ['Golden Grove', 'Boddington', 'Telfer'],
    'lon': [117.2, 116.4, 122.2],
    'lat': [-28.0, -32.8, -21.7],
    'commodity': ['Cu-Zn', 'Au-Cu', 'Au-Cu'],
    'resource_mt': [12.5, 250.0, 45.0]
})

deposits = spark.createDataFrame(deposits_df)
deposits = deposits.selectExpr(
    "*",
    "ST_Point(CAST(lon AS DOUBLE), CAST(lat AS DOUBLE)) AS geometry"
)

deposits.write.format("delta").mode("overwrite").saveAsTable("bronze.known_deposits")
# print(" Loaded known deposits")

# ======================================================================
# Code Block 6
# ======================================================================

# Create 50 km buffer around known deposits
deposits_buffered = spark.sql("""
    SELECT 
        deposit_id,
        name,
        commodity,
        ST_Buffer(geometry, 0.5) AS buffer_geom
    FROM bronze.known_deposits
""")

deposits_buffered.createOrReplaceTempView("deposits_buffered")

# Find clusters within 50 km of known deposits
clusters_near_deposits = spark.sql("""
    SELECT 
        c.cluster_id,
        c.n_samples,
        c.avg_Au,
        c.avg_Cu,
        c.avg_Zn,
        d.name AS nearest_deposit,
        d.commodity AS deposit_type,
        ST_Distance(ST_Centroid(c.cluster_polygon), d.buffer_geom) * 111 AS distance_km
    FROM (
        SELECT cluster_id, n_samples, avg_Au, avg_Cu, avg_Zn, cluster_polygon
        FROM gold.clustered_samples
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id, n_samples, avg_Au, avg_Cu, avg_Zn, cluster_polygon
    ) c
    CROSS JOIN (
        SELECT name, commodity, buffer_geom
        FROM deposits_buffered
    ) d
    WHERE ST_Intersects(ST_Centroid(c.cluster_polygon), d.buffer_geom)
""")

clusters_near_deposits.show(truncate=False)

# ======================================================================
# Code Block 7
# ======================================================================

# Classify clusters by geochemical signature
cluster_classification = spark.sql("""
    SELECT 
        cluster_id,
        n_samples,
        avg_Au,
        avg_Cu,
        avg_Zn,
        CASE
            WHEN avg_Au > 0.5 AND avg_Cu > 1000 THEN 'Porphyry Au-Cu'
            WHEN avg_Cu > 1000 AND avg_Zn < 500 THEN 'Porphyry Cu'
            WHEN avg_Zn > 1000 AND avg_Cu > 500 THEN 'VMS Cu-Zn'
            WHEN avg_Zn > 1000 THEN 'VMS Zn-Pb'
            WHEN avg_Au > 0.3 THEN 'Orogenic Au'
            ELSE 'Undefined'
        END AS deposit_class,
        cluster_polygon
    FROM (
        SELECT 
            cluster_id,
            COUNT(*) AS n_samples,
            AVG(Au_ppm) AS avg_Au,
            AVG(Cu_ppm) AS avg_Cu,
            AVG(Zn_ppm) AS avg_Zn,
            ST_ConvexHull(ST_Collect(geometry)) AS cluster_polygon
        FROM gold.clustered_samples
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
    )
    ORDER BY n_samples DESC
""")

cluster_classification.show(truncate=False)

# ======================================================================
# Code Block 8
# ======================================================================

# %pip install databricks-mosaic

import mosaic as mos
mos.enable_mosaic(spark, dbutils)

# ======================================================================
# Code Block 9
# ======================================================================

# %%mosaic_kepler

SELECT 
cluster_id,
deposit_class,
n_samples,
avg_Au,
# cluster_polygon AS geometry
# FROM gold.cluster_classification
# WHERE cluster_id IS NOT NULL

# ======================================================================
# Code Block 10
# ======================================================================

# Export clusters to GeoJSON
cluster_classification.selectExpr(
    "cluster_id",
    "deposit_class",
    "n_samples",
    "ST_AsText(cluster_polygon) AS wkt_geometry"
).write.format("json").mode("overwrite").save("/mnt/geo/clusters.geojson")

# print(" Exported to GeoJSON")

# ======================================================================
# Code Block 11
# ======================================================================

# Complete ore deposit clustering pipeline with Apache Sedona

from sedona.spark import *
from sedona.register import SedonaRegistrator
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
import pandas as pd

# ============================================================================
# 1. Initialize Sedona
# ============================================================================

spark = (SparkSession.builder
    .appName("DepositClustering")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
    .getOrCreate())

SedonaRegistrator.registerAll(spark)
# print(" Sedona initialized")

# ============================================================================
# 2. Generate synthetic geochemical data
# ============================================================================

np.random.seed(42)
n_samples = 10000

lons = np.random.uniform(115.0, 129.0, n_samples)
lats = np.random.uniform(-35.0, -16.0, n_samples)

cluster_centers = [(120.5, -30.5), (124.2, -22.8), (117.3, -28.1)]
cluster_labels = []

for lon, lat in zip(lons, lats):
    dists = [np.sqrt((lon - cx)**2 + (lat - cy)**2) for cx, cy in cluster_centers]
    nearest_idx = np.argmin(dists)
    cluster_labels.append(nearest_idx if dists[nearest_idx] < 2.0 else -1)

cluster_labels = np.array(cluster_labels)

Au_ppm = np.array([max(0.001, np.random.lognormal(-3 if c==-1 else 0, 1)) for c in cluster_labels])
Cu_ppm = np.where(cluster_labels == 1, Au_ppm * 3000, np.random.lognormal(1, 1, n_samples))
Zn_ppm = np.where(cluster_labels == 2, Au_ppm * 5000, np.random.lognormal(2, 1, n_samples))

geochem_df = pd.DataFrame({
    'sample_id': [f'GS{i:06d}' for i in range(n_samples)],
    'longitude': lons,
    'latitude': lats,
    'Au_ppm': Au_ppm,
    'Cu_ppm': Cu_ppm,
    'Zn_ppm': Zn_ppm
})

geochem = spark.createDataFrame(geochem_df)
geochem.write.format("delta").mode("overwrite").saveAsTable("bronze.geochem")
# print(f" Generated {n_samples:,} samples")

# ============================================================================
# 3. Create spatial points
# ============================================================================

samples = spark.sql("""
    SELECT 
        sample_id,
        longitude,
        latitude,
        Au_ppm,
        Cu_ppm,
        Zn_ppm,
        ST_Point(CAST(longitude AS DOUBLE), CAST(latitude AS DOUBLE)) AS geometry
    FROM bronze.geochem
    WHERE Au_ppm > 0.1 OR Cu_ppm > 100 OR Zn_ppm > 200
""")

samples.write.format("delta").mode("overwrite").saveAsTable("silver.anomalous")
# print(f" Filtered to {samples.count():,} anomalous samples")

# ============================================================================
# 4. Spatial clustering with DBSCAN
# ============================================================================

clustered = spark.sql("""
    SELECT 
        *,
        ST_ClusterDBSCAN(geometry, 0.1, 10) OVER() AS cluster_id
    FROM silver.anomalous
""")

clustered.write.format("delta").mode("overwrite").saveAsTable("gold.clustered")
# print(" DBSCAN complete")

# ============================================================================
# 5. Cluster statistics
# ============================================================================

stats = spark.sql("""
    SELECT 
        cluster_id,
        COUNT(*) AS n_samples,
        AVG(Au_ppm) AS avg_Au,
        AVG(Cu_ppm) AS avg_Cu,
        AVG(Zn_ppm) AS avg_Zn,
        MAX(Au_ppm) AS max_Au,
        CASE
            WHEN AVG(Au_ppm) > 0.5 AND AVG(Cu_ppm) > 1000 THEN 'Porphyry Au-Cu'
            WHEN AVG(Cu_ppm) > 1000 AND AVG(Zn_ppm) < 500 THEN 'Porphyry Cu'
            WHEN AVG(Zn_ppm) > 1000 AND AVG(Cu_ppm) > 500 THEN 'VMS Cu-Zn'
            WHEN AVG(Zn_ppm) > 1000 THEN 'VMS Zn-Pb'
            WHEN AVG(Au_ppm) > 0.3 THEN 'Orogenic Au'
            ELSE 'Undefined'
        END AS deposit_class
    FROM gold.clustered
    WHERE cluster_id IS NOT NULL
    GROUP BY cluster_id
    ORDER BY n_samples DESC
""")

stats.show(10, truncate=False)
# print("\n Analysis complete")

# ======================================================================
# Code Block 12
# ======================================================================

dists = [np.sqrt((lon - cx)**2 + (lat - cy)**2) for cx, cy in cluster_centers]
nearest_idx = np.argmin(dists)
nearest_dist = dists[nearest_idx]

# ======================================================================
# Code Block 13
# ======================================================================

if cluster_id == -1:  # Background
    return max(0.001, np.random.lognormal(-3, 1))
elif cluster_id == 0:  # Gold cluster
    return max(0.01, np.random.lognormal(0, 0.8))
elif cluster_id == 1:  # Copper-gold cluster
    return max(0.01, np.random.lognormal(-0.5, 0.6))
else:  # Zinc-lead cluster
    return max(0.01, np.random.lognormal(-1, 0.5))

# ======================================================================
# Code Block 14
# ======================================================================

# 'sample_id': [f'GS{i:06d}' for i in range(n_samples)],
# 'longitude': lons,
# 'latitude': lats,
# 'Au_ppm': Au_ppm,
# 'Cu_ppm': Cu_ppm,
# 'Zn_ppm': Zn_ppm,
# 'sample_type': np.random.choice(['soil', 'stream_sed', 'rock'], n_samples, p=[0.7, 0.2, 0.1])

# ======================================================================
# Code Block 15
# ======================================================================

# +----------+---------+-------+--------+--------+------------------+
# |cluster_id|n_samples|avg_Au |avg_Cu  |avg_Zn  |deposit_class     |
# +----------+---------+-------+--------+--------+------------------+
# |0         |1247     |0.845  |1834.2  |187.3   |Porphyry Au-Cu    |
# |1         |892      |0.234  |2847.6  |176.8   |Porphyry Cu       |
# |2         |531      |0.187  |102.4   |1872.4  |VMS Zn-Pb         |
# |3         |89       |0.456  |234.6   |198.1   |Orogenic Au       |
# +----------+---------+-------+--------+--------+------------------+

# ======================================================================
# Code Block 16
# ======================================================================

dists = [np.sqrt((lon - cx)**2 + (lat - cy)**2) for cx, cy in cluster_centers]
nearest_idx = np.argmin(dists)
cluster_labels.append(nearest_idx if dists[nearest_idx] < 2.0 else -1)
