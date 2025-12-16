#!/usr/bin/env python3
"""
Python code extracted from 25_dendrogram_three_projects_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

features = [wall_loss, cp_potential, soil_resistivity, coating_condition, proximity_to_water]
# Cluster segments with similar *combinations* of these features

# ======================================================================
# Code Block 2
# ======================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Synthetic demo data (in production, query from Delta tables)
np.random.seed(42)
N = 200

df = pd.DataFrame({
    'segment_id': [f'SEG-{i:04d}' for i in range(N)],
    'avg_wall_loss_pct': np.random.normal(15, 8, N).clip(0, 70),
    'cp_potential_mv': np.random.normal(-900, 50, N),
    'soil_resistivity_ohm_cm': np.random.normal(3000, 600, N).clip(200, 8000),
    'coating': np.random.choice(['FBE', 'PE', 'CoalTar', 'Tape'], N, p=[0.4, 0.3, 0.2, 0.1]),
    'near_water': np.random.choice([0, 1], N, p=[0.8, 0.2])
})

# Encode categorical: coating condition score
coating_score = {'FBE': 1.0, 'PE': 0.75, 'CoalTar': 0.5, 'Tape': 0.25}
df['coating_score'] = df['coating'].map(coating_score)

# ======================================================================
# Code Block 3
# ======================================================================

# Select numeric features
features = ['avg_wall_loss_pct', 'cp_potential_mv', 'soil_resistivity_ohm_cm', 
            'coating_score', 'near_water']
X = df[features].values

# Normalize (critical for features with different scales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute linkage using Ward's method
Z = linkage(X_scaled, method='ward')

# Visualize dendrogram
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(12, 6))

dendrogram(Z, ax=ax, truncate_mode='level', p=5, color_threshold=8,
           above_threshold_color='gray', no_labels=True)

ax.set_xlabel('Segment Group (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Pipeline Health Signature Dendrogram', fontsize=12, pad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('pipeline_health_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 4
# ======================================================================

# Cut dendrogram to form 5 clusters
n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df['cluster_id'] = clustering.fit_predict(X_scaled)

# Compute cluster profiles
cluster_profiles = df.groupby('cluster_id')[features].mean()
print(cluster_profiles.round(2))

# ======================================================================
# Code Block 5
# ======================================================================

import pandas as pd
import numpy as np

# Simulate 30 days of hourly SCADA data
T = 24 * 30
time = pd.date_range('2024-07-01', periods=T, freq='H')
np.random.seed(77)

# Realistic patterns: diurnal cycles + noise
flow = 200 + 10 * np.sin(2 * np.pi * time.hour / 24) + np.random.normal(0, 4, T)
p_suction = 600 + 5 * np.sin(2 * np.pi * time.hour / 12) + np.random.normal(0, 2, T)
p_discharge = 640 + 6 * np.sin(2 * np.pi * time.hour / 12) + np.random.normal(0, 3, T)

df_scada = pd.DataFrame({
    'timestamp': time,
    'flow_m3h': flow,
    'p_suction_kpa': p_suction,
    'p_discharge_kpa': p_discharge
})

# Compute rolling features per 24-hour window
window_size = 24
daily_features = []

for i in range(0, len(df_scada) - window_size, window_size):
    window = df_scada.iloc[i:i + window_size]
    
    daily_features.append({
        'day': i // window_size,
        'flow_mean': window['flow_m3h'].mean(),
        'flow_std': window['flow_m3h'].std(),
        'flow_kurtosis': window['flow_m3h'].kurtosis(),
        'dp_mean': (window['p_discharge_kpa'] - window['p_suction_kpa']).mean(),
        'dp_std': (window['p_discharge_kpa'] - window['p_suction_kpa']).std()
    })

df_daily = pd.DataFrame(daily_features)

# ======================================================================
# Code Block 6
# ======================================================================

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# Normalize features
features_ops = ['flow_mean', 'flow_std', 'flow_kurtosis', 'dp_mean', 'dp_std']
X_ops = df_daily[features_ops].values
X_ops_scaled = StandardScaler().fit_transform(X_ops)

# Hierarchical clustering
Z_ops = linkage(X_ops_scaled, method='ward')

# Dendrogram
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z_ops, ax=ax, truncate_mode='level', p=4, color_threshold=5,
           above_threshold_color='gray', no_labels=True)

ax.set_xlabel('Day Window (sorted by similarity)', fontsize=11)
ax.set_ylabel('Linkage Distance', fontsize=11)
ax.set_title('Compressor Operating Regimes Dendrogram', fontsize=12, pad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('compressor_regimes_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Assign clusters
df_daily['cluster_id'] = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(X_ops_scaled)

# ======================================================================
# Code Block 7
# ======================================================================

fig, ax = plt.subplots(figsize=(12, 4))

colors_regimes = ['#2ecc71', '#e67e22', '#95a5a6', '#e74c3c']
for day, cluster in enumerate(df_daily['cluster_id']):
    ax.bar(day, 1, color=colors_regimes[cluster], edgecolor='black', linewidth=0.3)

ax.set_xlabel('Day', fontsize=11)
ax.set_ylabel('Operational Regime', fontsize=11)
ax.set_title('Daily Operational Cluster Index Over Time', fontsize=12, pad=15)
ax.set_yticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('compressor_cluster_timeline.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 8
# ======================================================================

# Synthetic Sentinel-2 features (in production, computed with Databricks Mosaic)
np.random.seed(9)
N_tiles = 150

df_row = pd.DataFrame({
    'tile_id': range(1, N_tiles + 1),
    'chainage_km': np.linspace(0, 150, N_tiles),
    'ndvi_mean': np.random.uniform(0.1, 0.8, N_tiles),
    'ndvi_std': np.random.uniform(0.01, 0.2, N_tiles),
    'texture_glcm': np.random.uniform(0.05, 0.5, N_tiles),
    'thermal_anomaly_score': np.random.normal(0, 0.3, N_tiles),
    'bare_soil_fraction': np.random.uniform(0, 0.7, N_tiles)
})

# ======================================================================
# Code Block 9
# ======================================================================

features_row = ['ndvi_mean', 'ndvi_std', 'texture_glcm', 
                'thermal_anomaly_score', 'bare_soil_fraction']
X_row = df_row[features_row].values
X_row_scaled = StandardScaler().fit_transform(X_row)

Z_row = linkage(X_row_scaled, method='ward')

# Dendrogram
fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z_row, ax=ax, truncate_mode='level', p=6, color_threshold=4,
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

# Assign clusters
df_row['cluster_id'] = AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(X_row_scaled)

# ======================================================================
# Code Block 10
# ======================================================================

# In production, visualize with Mosaic
import mosaic as mos

# Save to Delta table
spark.createDataFrame(df_row).write.mode('overwrite').saveAsTable('gold.row_clusters')

# Visualize
df_spark = spark.table('gold.row_clusters')
mos.display(df_spark, geometry_col='geometry', color='cluster_id', 
            title='ROW Environmental Clusters Along Pipeline')

# ======================================================================
# Code Block 11
# ======================================================================

import dlt

@dlt.table(
    comment="Pipeline segment health clusters",
    table_properties={"quality": "gold"}
)
def segment_health_clusters():
    # Read silver-level features
    df_features = spark.table('silver.segment_features')
    
    # Convert to Pandas for scikit-learn clustering
    pdf = df_features.toPandas()
    
    # Clustering logic (same as above)
    X_scaled = StandardScaler().fit_transform(pdf[features])
    pdf['cluster_id'] = AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(X_scaled)
    
    # Return as Spark DataFrame
    return spark.createDataFrame(pdf[['segment_id', 'cluster_id']])

# ======================================================================
# Code Block 12
# ======================================================================

# Track clustering experiments with MLflow
import mlflow

with mlflow.start_run():
    mlflow.log_param('linkage_method', 'ward')
    mlflow.log_param('n_clusters', 5)
    mlflow.log_param('features', features)
    
    # Clustering code here
    
    mlflow.log_metric('silhouette_score', silhouette_score(X_scaled, labels))
    mlflow.log_artifact('dendrogram.png')

# ======================================================================
# Code Block 13
# ======================================================================

features_combined = [
    'avg_wall_loss_pct',          # ILI
    'cp_potential_mv',             # CP survey
    'pressure_variance_90d',       # SCADA
    'ndvi_mean',                   # Sentinel-2
    'thermal_anomaly_score'        # Sentinel-2
]

# ======================================================================
# Code Block 14
# ======================================================================

# Databricks Job: Run weekly, alert on cluster transitions
new_high_risk = spark.sql("""
SELECT segment_id FROM gold.cluster_history
WHERE clustering_date = CURRENT_DATE() AND cluster_id IN (2, 4)
  AND segment_id NOT IN (
    SELECT segment_id FROM gold.cluster_history
    WHERE clustering_date = DATE_SUB(CURRENT_DATE(), 7) AND cluster_id IN (2, 4)
  )
""")

if new_high_risk.count() > 0:
    # Send Slack alert
    dbutils.notebook.run('/Alerts/send_slack_message', 60, 
                         {'message': f'{new_high_risk.count()} segments moved to high-risk'})

# ======================================================================
# Code Block 15
# ======================================================================

'segment_id': [f'SEG-{i:04d}' for i in range(N)],
'avg_wall_loss_pct': np.random.normal(15, 8, N).clip(0, 70),
'cp_potential_mv': np.random.normal(-900, 50, N),
'soil_resistivity_ohm_cm': np.random.normal(3000, 600, N).clip(200, 8000),
'coating': np.random.choice(['FBE', 'PE', 'CoalTar', 'Tape'], N, p=[0.4, 0.3, 0.2, 0.1]),
'near_water': np.random.choice([0, 1], N, p=[0.8, 0.2])

# ======================================================================
# Code Block 16
# ======================================================================

above_threshold_color='gray', no_labels=True)

# ======================================================================
# Code Block 17
# ======================================================================

window = df_scada.iloc[i:i + window_size]

daily_features.append({
    'day': i // window_size,
    'flow_mean': window['flow_m3h'].mean(),
    'flow_std': window['flow_m3h'].std(),
    'flow_kurtosis': window['flow_m3h'].kurtosis(),
    'dp_mean': (window['p_discharge_kpa'] - window['p_suction_kpa']).mean(),
    'dp_std': (window['p_discharge_kpa'] - window['p_suction_kpa']).std()
})

# ======================================================================
# Code Block 18
# ======================================================================

above_threshold_color='gray', no_labels=True)

# ======================================================================
# Code Block 19
# ======================================================================

ax.bar(day, 1, color=colors_regimes[cluster], edgecolor='black', linewidth=0.3)

# ======================================================================
# Code Block 20
# ======================================================================

above_threshold_color='gray', no_labels=True)

# ======================================================================
# Code Block 21
# ======================================================================

title='ROW Environmental Clusters Along Pipeline')

# ======================================================================
# Code Block 22
# ======================================================================

comment="Pipeline segment health clusters",
table_properties={"quality": "gold"}

# ======================================================================
# Code Block 23
# ======================================================================

df_features = spark.table('silver.segment_features')

# ======================================================================
# Code Block 24
# ======================================================================

pdf = df_features.toPandas()

# ======================================================================
# Code Block 25
# ======================================================================

X_scaled = StandardScaler().fit_transform(pdf[features])
pdf['cluster_id'] = AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(X_scaled)

# ======================================================================
# Code Block 26
# ======================================================================

return spark.createDataFrame(pdf[['segment_id', 'cluster_id']])

# ======================================================================
# Code Block 27
# ======================================================================

- **Leak events:** 12 → 2 (83% reduction)
- **Annual inspection cost:** \$4.2M → \$3.1M (26% savings)
- **Avoided leak costs:** 10 leaks × \$850K = \$8.5M
- **Net ROI:** \$8.5M + 3 × \$1.1M - \$9.1M = **\$2.7M positive**

# ======================================================================
# Code Block 28
# ======================================================================

SELECT segment_id FROM gold.cluster_history
WHERE clustering_date = DATE_SUB(CURRENT_DATE(), 7) AND cluster_id IN (2, 4)
