#!/usr/bin/env python3
"""
Python code extracted from 26_row_vegetation_clustering_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

features = [ndvi_mean, ndvi_std, texture, thermal_anomaly, bare_soil_fraction]
# Group tiles by similarity across all features, not just NDVI

# ======================================================================
# Code Block 2
# ======================================================================

from pyspark.sql import SparkSession
import mosaic as mos

spark = SparkSession.builder.getOrCreate()
mos.enable_mosaic(spark, dbutils)
mos.enable_gdal(spark)  # For raster support

# ======================================================================
# Code Block 3
# ======================================================================

# Load Sentinel-2 GeoTIFFs from cloud storage
df_rasters = spark.read.format('gdal') \
    .option('raster_storage', 'dbfs:/sentinel2/ROW_tiles/') \
    .load()

df_rasters.createOrReplaceTempView('sentinel2_raw')

# ======================================================================
# Code Block 4
# ======================================================================

from pyspark.sql.functions import udf
from skimage.feature import graycomatrix, graycoprops
import numpy as np

@udf('double')
def compute_texture_glcm(nir_band):
    """Compute GLCM contrast metric from NIR band."""
    # nir_array = np.array(nir_band).reshape(100, 100)  # Assuming 100100 pixels per tile
    nir_normalized = ((nir_array - nir_array.min()) / (nir_array.max() - nir_array.min()) * 255).astype(np.uint8)
    
    glcm = graycomatrix(nir_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    return float(contrast)

df_texture = df_rasters.withColumn('texture_glcm', compute_texture_glcm('band_nir'))

# ======================================================================
# Code Block 5
# ======================================================================

# Bare soil: NDVI < 0.2
df_features = spark.sql("""
SELECT
    tile_id,
    ndvi_mean,
    ndvi_std,
    SUM(CASE WHEN ndvi < 0.2 THEN 1 ELSE 0 END) / COUNT(*) AS bare_soil_fraction
FROM (
    SELECT tile_id, (band_nir - band_red) / (band_nir + band_red) AS ndvi
    FROM sentinel2_raw
)
GROUP BY tile_id, ndvi_mean, ndvi_std
""")

# ======================================================================
# Code Block 6
# ======================================================================

import pandas as pd
import numpy as np

np.random.seed(9)
N_tiles = 150

df_row = pd.DataFrame({
    'tile_id': range(1, N_tiles + 1),
    'chainage_km': np.linspace(0, 150, N_tiles),
    'longitude': np.random.uniform(-110.5, -109.5, N_tiles),
    'latitude': np.random.uniform(35.0, 36.0, N_tiles),
    'ndvi_mean': np.random.uniform(0.1, 0.8, N_tiles),
    'ndvi_std': np.random.uniform(0.01, 0.2, N_tiles),
    'texture_glcm': np.random.uniform(0.05, 0.5, N_tiles),
    'thermal_anomaly_score': np.random.normal(0, 0.3, N_tiles),
    'bare_soil_fraction': np.random.uniform(0, 0.7, N_tiles)
})

# Create realistic patterns
# Cluster 0: Dense vegetation (forest)
forest_indices = np.random.choice(N_tiles, size=40, replace=False)
df_row.loc[forest_indices, 'ndvi_mean'] = np.random.uniform(0.65, 0.80, 40)
df_row.loc[forest_indices, 'ndvi_std'] = np.random.uniform(0.10, 0.20, 40)
df_row.loc[forest_indices, 'bare_soil_fraction'] = np.random.uniform(0.0, 0.10, 40)

# Cluster 1: Bare soil (recent clearing or disturbance)
bare_indices = np.random.choice([i for i in range(N_tiles) if i not in forest_indices], size=25, replace=False)
df_row.loc[bare_indices, 'ndvi_mean'] = np.random.uniform(0.1, 0.25, 25)
df_row.loc[bare_indices, 'bare_soil_fraction'] = np.random.uniform(0.60, 0.85, 25)
df_row.loc[bare_indices, 'thermal_anomaly_score'] = np.random.uniform(0.3, 0.8, 25)

# ======================================================================
# Code Block 7
# ======================================================================

from sklearn.preprocessing import StandardScaler

features = ['ndvi_mean', 'ndvi_std', 'texture_glcm', 
            'thermal_anomaly_score', 'bare_soil_fraction']
X = df_row[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature statistics:")
print(pd.DataFrame(X_scaled, columns=features).describe())

# ======================================================================
# Code Block 8
# ======================================================================

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Ward linkage minimizes within-cluster variance
Z = linkage(X_scaled, method='ward')

# Visualize dendrogram
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 5))

dendrogram(Z, ax=ax, truncate_mode='level', p=6, color_threshold=4,
           above_threshold_color='gray', no_labels=True)

ax.set_xlabel('Tile Index (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Right-of-Way Environmental Clusters Dendrogram', fontsize=12, pad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('row_vegetation_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 9
# ======================================================================

from sklearn.cluster import AgglomerativeClustering

n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df_row['cluster_id'] = clustering.fit_predict(X_scaled)

# Compute cluster profiles
cluster_profiles = df_row.groupby('cluster_id')[features].mean()
cluster_counts = df_row.groupby('cluster_id').size()

print("\nCluster Profiles:")
print(cluster_profiles.round(3))
print("\nCluster Sizes:")
print(cluster_counts)

# ======================================================================
# Code Block 10
# ======================================================================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))

colors_clusters = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#e67e22']
cluster_names = ['Dense Veg', 'Bare/Disturbed', 'Mixed Cover', 'Healthy', 'Thermal Anomaly']

for i in range(n_clusters):
    cluster_data = df_row[df_row['cluster_id'] == i]
    ax.scatter(cluster_data['chainage_km'], cluster_data['ndvi_mean'],
               c=colors_clusters[i], label=f'C{i}: {cluster_names[i]}',
               s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Chainage (km)', fontsize=11)
ax.set_ylabel('NDVI (0-1)', fontsize=11)
ax.set_title('ROW Vegetation Clusters by Location', fontsize=12, pad=15)
ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
ax.grid(False)

plt.tight_layout()
plt.savefig('row_clusters_spatial.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 11
# ======================================================================

import mosaic as mos

# Convert Pandas to Spark DataFrame
df_spark = spark.createDataFrame(df_row)

# Create geometries from coordinates
df_spark = df_spark.selectExpr(
    '*',
    'ST_Point(longitude, latitude) AS geometry'
)

# Save to Delta Gold table
df_spark.write.mode('overwrite').saveAsTable('gold.row_clusters')

# Visualize with Mosaic
df_map = spark.table('gold.row_clusters')
mos.display(df_map, geometry_col='geometry', color='cluster_id', 
            title='ROW Environmental Clusters', 
            legend_title='Cluster ID')

# ======================================================================
# Code Block 12
# ======================================================================

# Compute NDVI trend over last 6 months
features_temporal = [
    'ndvi_mean_current',
    'ndvi_mean_30d_ago',
    'ndvi_mean_90d_ago',
    'ndvi_trend_slope',  # Linear regression slope
    'ndvi_seasonality'   # STD of monthly means
]

# ======================================================================
# Code Block 13
# ======================================================================

features_multi_sensor = [
    'ndvi_mean',                  # Sentinel-2
    'sar_coherence',              # Sentinel-1
    'sar_vv_backscatter',         # Sentinel-1
    'thermal_anomaly',            # Sentinel-2
    'bare_soil_fraction'          # Sentinel-2
]

# ======================================================================
# Code Block 14
# ======================================================================

# Databricks Job: Run weekly
high_risk_tiles = spark.sql("""
SELECT tile_id, chainage_km, cluster_id, ndvi_mean
FROM gold.row_clusters
WHERE cluster_id IN (1, 4)  -- Disturbed or Thermal Anomaly
  AND last_patrol_date < DATE_SUB(CURRENT_DATE(), 7)
""")

if high_risk_tiles.count() > 0:
    # Generate work orders in CMMS
    work_orders = high_risk_tiles.toPandas().to_dict('records')
    for wo in work_orders:
        dbutils.notebook.run('/WorkOrders/create_patrol_order', 60, wo)

# ======================================================================
# Code Block 15
# ======================================================================

import mosaic as mos
mos.enable_mosaic(spark, dbutils)
mos.enable_gdal(spark)

# ======================================================================
# Code Block 16
# ======================================================================

# Databricks Notebook: ROW Vegetation Clustering
# Prerequisites:
# 1. Sentinel-2 GeoTIFFs in dbfs:/sentinel2/ROW_tiles/
# 2. Mosaic enabled on cluster

# COMMAND ----------
# Setup
# %pip install -q scipy scikit-learn matplotlib pandas
dbutils.library.restartPython()

# COMMAND ----------
# Import libraries
from pyspark.sql import SparkSession
import mosaic as mos
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()
mos.enable_mosaic(spark, dbutils)

# COMMAND ----------
# Ingest Sentinel-2 (real data)
# df_rasters = spark.read.format('gdal') \
#     .option('raster_storage', 'dbfs:/sentinel2/ROW_tiles/') \
#     .load()

# For demo: use synthetic data
np.random.seed(9)
N = 150

data = {
    'tile_id': range(1, N + 1),
    'chainage_km': np.linspace(0, 150, N),
    'longitude': np.random.uniform(-110.5, -109.5, N),
    'latitude': np.random.uniform(35.0, 36.0, N),
    'ndvi_mean': np.random.uniform(0.1, 0.8, N),
    'ndvi_std': np.random.uniform(0.01, 0.2, N),
    'texture_glcm': np.random.uniform(0.05, 0.5, N),
    'thermal_anomaly_score': np.random.normal(0, 0.3, N),
    'bare_soil_fraction': np.random.uniform(0, 0.7, N)
}

df_row = pd.DataFrame(data)
print(f'Loaded {len(df_row)} tiles')

# COMMAND ----------
# Normalize features
features = ['ndvi_mean', 'ndvi_std', 'texture_glcm', 
            'thermal_anomaly_score', 'bare_soil_fraction']
X = df_row[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(' Features normalized')

# COMMAND ----------
# Hierarchical clustering
Z = linkage(X_scaled, method='ward')
# print(' Linkage computed')

# COMMAND ----------
# Dendrogram
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 5))

dendrogram(Z, ax=ax, truncate_mode='level', p=6, color_threshold=4,
           above_threshold_color='gray', no_labels=True)

ax.set_xlabel('Tile Index', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('ROW Environmental Clusters', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('/dbfs/FileStore/row_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()
# print(' Dendrogram saved')

# COMMAND ----------
# Assign clusters
n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df_row['cluster_id'] = clustering.fit_predict(X_scaled)

cluster_profiles = df_row.groupby('cluster_id')[features].mean()
print('\nCluster Profiles:')
print(cluster_profiles.round(3))

# COMMAND ----------
# Save to Delta
df_spark = spark.createDataFrame(df_row)
df_spark = df_spark.selectExpr('*', 'ST_Point(longitude, latitude) AS geometry')
df_spark.write.mode('overwrite').saveAsTable('gold.row_clusters')
# print(' Saved to gold.row_clusters')

# COMMAND ----------
# Spatial visualization
fig, ax = plt.subplots(figsize=(12, 4))

colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#e67e22']
names = ['Dense Veg', 'Bare/Disturbed', 'Mixed', 'Healthy', 'Thermal']

for i in range(n_clusters):
    data = df_row[df_row['cluster_id'] == i]
    ax.scatter(data['chainage_km'], data['ndvi_mean'],
               c=colors[i], label=f'C{i}: {names[i]}',
               s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Chainage (km)', fontsize=11)
ax.set_ylabel('NDVI', fontsize=11)
ax.set_title('ROW Clusters by Location', fontsize=12, pad=15)
ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('/dbfs/FileStore/row_spatial.png', dpi=300, bbox_inches='tight')
plt.show()
# print(' Spatial map saved')

# ======================================================================
# Code Block 17
# ======================================================================

"""Compute GLCM contrast metric from NIR band."""
# nir_array = np.array(nir_band).reshape(100, 100)  # Assuming 100100 pixels per tile
nir_normalized = ((nir_array - nir_array.min()) / (nir_array.max() - nir_array.min()) * 255).astype(np.uint8)

glcm = graycomatrix(nir_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
contrast = graycoprops(glcm, 'contrast')[0, 0]
return float(contrast)

# ======================================================================
# Code Block 18
# ======================================================================

# above_threshold_color='gray', no_labels=True

# ======================================================================
# Code Block 19
# ======================================================================

cluster_data = df_row[df_row['cluster_id'] == i]
ax.scatter(cluster_data['chainage_km'], cluster_data['ndvi_mean'],
           c=colors_clusters[i], label=f'C{i}: {cluster_names[i]}',
           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

# ======================================================================
# Code Block 20
# ======================================================================

title='ROW Environmental Clusters', 
legend_title='Cluster ID'

# ======================================================================
# Code Block 21
# ======================================================================

SELECT
curr.tile_id,
curr.chainage_km,
# prev.cluster_id AS prev_cluster,
# curr.cluster_id AS curr_cluster,
# curr.ndvi_mean - prev.ndvi_mean AS ndvi_change
# FROM gold.row_cluster_history curr
# JOIN gold.row_cluster_history prev
# ON curr.tile_id = prev.tile_id
# AND prev.clustering_date = DATE_SUB(curr.clustering_date, 30)  -- 30 days ago
# WHERE curr.clustering_date = CURRENT_DATE()

# ======================================================================
# Code Block 22
# ======================================================================

work_orders = high_risk_tiles.toPandas().to_dict('records')
for wo in work_orders:
    dbutils.notebook.run('/WorkOrders/create_patrol_order', 60, wo)

# ======================================================================
# Code Block 23
# ======================================================================

# above_threshold_color='gray', no_labels=True

# ======================================================================
# Code Block 24
# ======================================================================

data = df_row[df_row['cluster_id'] == i]
ax.scatter(data['chainage_km'], data['ndvi_mean'],
           c=colors[i], label=f'C{i}: {names[i]}',
           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
