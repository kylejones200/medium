#!/usr/bin/env python3
"""
Python code extracted from 23_pipeline_health_clustering_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

# Corrosion rate (requires historical ILI data)
corrosion_rate = (wall_loss_current - wall_loss_previous) / years_between_ili

# CP adequacy flag
cp_adequate = (avg_cp_potential_mv < -850)  # Typical criterion

# Risk score (simple weighted combination)
risk_score = (
    0.4 * max_wall_loss_pct +
    0.3 * (1 - cp_adequate) * 100 +
    0.2 * np.log1p(soil_resistivity_ohm_cm / 1000) +
    0.1 * anomaly_count
)

# ======================================================================
# Code Block 2
# ======================================================================

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load segment features from Delta table
df = spark.table('silver.segment_features').toPandas()

# Select numeric features for clustering
features = [
    'avg_wall_loss_pct',
    'max_wall_loss_pct',
    'anomaly_count',
    'critical_anomaly_count',
    'avg_cp_potential_mv',
    'cp_std_mv',
    'avg_soil_resistivity_ohm_cm',
    'years_since_last_ili',
    'operating_pressure_mpa'
]

X = df[features].fillna(df[features].median())

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute linkage (Ward minimizes within-cluster variance)
Z = linkage(X_scaled, method='ward')

# Cut dendrogram at height that yields ~5 clusters
cluster_labels = fcluster(Z, t=5, criterion='maxclust')
df['cluster_id'] = cluster_labels

# Save to Delta
result_df = spark.createDataFrame(df[['segment_id', 'cluster_id']])
result_df.write.mode('overwrite').saveAsTable('gold.segment_clusters')

# ======================================================================
# Code Block 3
# ======================================================================

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 6))

dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-5, 2])

ax.set_xlabel('Segment Index (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Pipeline Segment Hierarchical Clustering Dendrogram', fontsize=12, pad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('pipeline_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 4
# ======================================================================

# Compute mean feature values per cluster
cluster_profiles = df.groupby('cluster_id')[features].mean()
cluster_profiles['segment_count'] = df.groupby('cluster_id').size()

print(cluster_profiles.round(2))

# ======================================================================
# Code Block 5
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

# Assuming df has 'start_chainage_km' for spatial location
fig, ax = plt.subplots(figsize=(12, 4))

# Color map for clusters
colors = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', '#e74c3c']
cluster_names = ['Healthy', 'Moderate', 'High Risk', 'Stable', 'Critical']

for i, cluster_id in enumerate(range(1, 6)):
    cluster_data = df[df['cluster_id'] == cluster_id]
    ax.scatter(cluster_data['start_chainage_km'], 
               cluster_data['max_wall_loss_pct'],
               c=colors[i], label=f'C{cluster_id}: {cluster_names[i]}',
               s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

ax.set_xlabel('Chainage (km)', fontsize=11)
ax.set_ylabel('Max Wall Loss (%)', fontsize=11)
ax.set_title('Pipeline Segment Clusters by Location and Wall Loss', fontsize=12, pad=15)
ax.legend(loc='upper left', frameon=False, fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('pipeline_clusters_spatial.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 6
# ======================================================================

# Rolling 24-hour window features
features_operational = [
    'mean_suction_pressure_mpa',
    'std_suction_pressure_mpa',
    'mean_discharge_pressure_mpa',
    'std_discharge_pressure_mpa',
    'mean_flow_rate_m3h',
    'flow_rate_kurtosis',  # Detects spikes
    'pressure_ratio_mean',
    'vibration_rms_avg'
]

# ======================================================================
# Code Block 7
# ======================================================================

# Databricks Notebook: Pipeline Health Clustering
# Prereqs: ILI, CP, and soil data in bronze tables

# COMMAND ----------
# Install dependencies
# %pip install -q scipy scikit-learn matplotlib pandas
dbutils.library.restartPython()

# COMMAND ----------
# Configuration
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

spark = SparkSession.builder.getOrCreate()

CATALOG = 'pipeline'
SCHEMA = 'integrity'
TABLE_FEATURES = f'{CATALOG}.{SCHEMA}.segment_features'
TABLE_CLUSTERS = f'{CATALOG}.{SCHEMA}.segment_clusters'

# COMMAND ----------
# Feature engineering (run this in SQL notebook cell or via spark.sql)
spark.sql(f"""
CREATE OR REPLACE TABLE {TABLE_FEATURES} AS
SELECT
  s.segment_id,
  s.segment_length_m,
  s.operating_pressure_mpa,
  s.coating_condition,
  s.years_since_last_ili,
  ili.avg_wall_loss_pct,
  ili.max_wall_loss_pct,
  ili.anomaly_count,
  ili.critical_anomaly_count,
  cp.avg_cp_potential_mv,
  cp.cp_std_mv,
  cp.min_cp_potential_mv,
  soil.avg_soil_resistivity_ohm_cm,
  s.start_chainage_km
FROM {CATALOG}.bronze.segments s
LEFT JOIN {CATALOG}.silver.segment_ili_features ili ON s.segment_id = ili.segment_id
LEFT JOIN {CATALOG}.silver.segment_cp_features cp ON s.segment_id = cp.segment_id
LEFT JOIN {CATALOG}.silver.segment_soil_features soil ON s.segment_id = soil.segment_id
WHERE ili.avg_wall_loss_pct IS NOT NULL
""")

# print(f' Feature table created: {TABLE_FEATURES}')

# COMMAND ----------
# Load features
df = spark.table(TABLE_FEATURES).toPandas()
print(f'Loaded {len(df):,} segments')

# Select numeric features
features = [
    'avg_wall_loss_pct',
    'max_wall_loss_pct',
    'anomaly_count',
    'critical_anomaly_count',
    'avg_cp_potential_mv',
    'cp_std_mv',
    'avg_soil_resistivity_ohm_cm',
    'years_since_last_ili',
    'operating_pressure_mpa'
]

X = df[features].fillna(df[features].median())

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(' Features normalized')

# COMMAND ----------
# Hierarchical clustering
Z = linkage(X_scaled, method='ward')
# print(' Linkage computed')

# Cut dendrogram to form 5 clusters
n_clusters = 5
cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
df['cluster_id'] = cluster_labels

# Save to Delta
result_df = spark.createDataFrame(df[['segment_id', 'cluster_id']])
result_df.write.mode('overwrite').saveAsTable(TABLE_CLUSTERS)
# print(f' Clusters saved to {TABLE_CLUSTERS}')

# COMMAND ----------
# Dendrogram visualization
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 6))

dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-n_clusters, 2])

ax.set_xlabel('Segment Index', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Pipeline Segment Hierarchical Clustering', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('/dbfs/FileStore/pipeline_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()
# print(' Dendrogram saved')

# COMMAND ----------
# Cluster profiling
cluster_profiles = df.groupby('cluster_id')[features].mean()
cluster_profiles['segment_count'] = df.groupby('cluster_id').size()
print('\nCluster Profiles:')
print(cluster_profiles.round(2))

# COMMAND ----------
# Spatial visualization
fig, ax = plt.subplots(figsize=(12, 4))

colors = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', '#e74c3c']
cluster_names = ['Healthy', 'Moderate', 'High Risk', 'Stable', 'Critical']

for i, cluster_id in enumerate(range(1, n_clusters + 1)):
    cluster_data = df[df['cluster_id'] == cluster_id]
    ax.scatter(cluster_data['start_chainage_km'], 
               cluster_data['max_wall_loss_pct'],
               c=colors[i], label=f'C{cluster_id}: {cluster_names[i]}',
               s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

ax.set_xlabel('Chainage (km)', fontsize=11)
ax.set_ylabel('Max Wall Loss (%)', fontsize=11)
ax.set_title('Cluster Distribution Along Pipeline', fontsize=12, pad=15)
ax.legend(loc='upper left', frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('/dbfs/FileStore/pipeline_clusters_spatial.png', dpi=300, bbox_inches='tight')
plt.show()
# print(' Spatial map saved')

# ======================================================================
# Code Block 8
# ======================================================================

cluster_data = df[df['cluster_id'] == cluster_id]
ax.scatter(cluster_data['start_chainage_km'], 
           cluster_data['max_wall_loss_pct'],
           c=colors[i], label=f'C{cluster_id}: {cluster_names[i]}',
           s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

# ======================================================================
# Code Block 9
# ======================================================================

# - **Cluster 3:** Installed 15 new CP rectifiers at \$45K each =
  # \$675K.
# - **Cluster 5:** Replaced 70 worst segments at \$120K each = \$8.4M
# (one-time).

# ======================================================================
# Code Block 10
# ======================================================================

# - **Leak events:** 12 → 2 (83% reduction).
# - **Annual inspection budget:** \$4.2M → \$3.1M (26% reduction).
# - **Avoided leak costs:** 10 leaks  \$850K = \$8.5M saved.
# - **Net savings over 3 years:** \$8.5M + 3  \$1.1M - \$9.1M =
# **\$2.7M positive ROI**.

# ======================================================================
# Code Block 11
# ======================================================================

# ON curr.segment_id = prev.segment_id
# AND prev.clustering_date = DATE_SUB(curr.clustering_date, 365)

# ======================================================================
# Code Block 12
# ======================================================================

cluster_data = df[df['cluster_id'] == cluster_id]
ax.scatter(cluster_data['start_chainage_km'], 
           cluster_data['max_wall_loss_pct'],
           c=colors[i], label=f'C{cluster_id}: {cluster_names[i]}',
           s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
