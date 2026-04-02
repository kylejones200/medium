#!/usr/bin/env python3
"""
Python code extracted from 27_thirdparty_detection_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import numpy as np

# Define pipeline ROW as series of lat/lon waypoints
row_centerline = pd.DataFrame({
    'node_id': range(60),
    'lat': 29.5 + np.linspace(0, 1.2, 60),
    'lon': -95.5 + 0.6 * np.sin(np.linspace(0, 3.14, 60)),
    'chainage_km': np.linspace(0, 150, 60)
})

# Save to Delta for spatial joins
spark.createDataFrame(row_centerline).write.mode('overwrite').saveAsTable('reference.row_centerline')

# ======================================================================
# Code Block 2
# ======================================================================

import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points in kilometers.
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = (np.sin(dlat / 2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# Register as Spark UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

haversine_udf = udf(haversine_distance, DoubleType())
spark.udf.register('haversine', haversine_udf)

# ======================================================================
# Code Block 3
# ======================================================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load events with distance
df = spark.table('silver.row_events_with_distance').toPandas()

# Build scoring components
components = pd.DataFrame({
    'distance_km': df['dist_to_row_km'],
    'days_old': df['days_since_event'],
    'no_watch': (~df['has_active_watch']).astype(float),
    'is_machinery': (df['event_type'] == 'gps_equipment').astype(float)
})

# Normalize to [0, 1] range
scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(components),
    columns=components.columns,
    index=components.index
)

# Compute weighted risk score
risk_score = (
    1.0 * (1 - normalized['distance_km']) +    # Closer = higher risk (40%)
    1.0 * (1 - normalized['days_old']) +       # More recent = higher risk (30%)
    0.8 * normalized['no_watch'] +             # No watch = higher risk (20%)
    0.7 * normalized['is_machinery']           # Active machinery = higher risk (10%)
)

df['risk_score'] = risk_score

# ======================================================================
# Code Block 4
# ======================================================================

# Databricks Job: Run every 15 minutes
high_risk_events = spark.sql("""
SELECT
    event_id,
    event_type,
    lat,
    lon,
    dist_to_row_km,
    days_since_event,
    has_active_watch,
    risk_score
FROM gold.row_risk_scores
WHERE risk_score >= 0.8
    AND alert_sent = FALSE
ORDER BY risk_score DESC
""")

if high_risk_events.count() > 0:
    # Send SMS/email to patrol team
    for row in high_risk_events.collect():
        send_alert(
            recipient='patrol_team@pipeline.com',
            subject=f'CRITICAL: Third-party work detected at {row.lat:.5f}, {row.lon:.5f}',
            body=f'''
Risk Score: {row.risk_score:.3f}
Event Type: {row.event_type}
Distance to ROW: {row.dist_to_row_km:.2f} km
Days Old: {row.days_since_event}
Active Watch: {row.has_active_watch}

Immediate field inspection required.
Google Maps: https://maps.google.com/?q={row.lat},{row.lon}
            '''
        )
    
    # Mark as alerted
    spark.sql(f"""
    UPDATE gold.row_risk_scores
    SET alert_sent = TRUE, alert_timestamp = CURRENT_TIMESTAMP()
    WHERE risk_score >= 0.8 AND alert_sent = FALSE
    """)

# ======================================================================
# Code Block 5
# ======================================================================

# Cluster events by spatial proximity and temporal clustering
from sklearn.cluster import DBSCAN
import numpy as np

# Features: lat, lon, days_old (normalized to spatial scale)
X = df[['lat', 'lon']].values
X_scaled = StandardScaler().fit_transform(X)

# DBSCAN: Find spatial clusters of events
clustering = DBSCAN(eps=0.05, min_samples=5)
df['spatial_cluster'] = clustering.fit_predict(X_scaled)

# Identify clusters with multiple event types (811 + permit + GPS = confirmed project)
cluster_summary = df.groupby('spatial_cluster').agg({
    'event_type': lambda x: x.nunique(),
    'event_id': 'count',
    'risk_score': 'mean'
})

high_activity_clusters = cluster_summary[
    (cluster_summary['event_type'] >= 2) &  # Multiple event types
    (cluster_summary['event_id'] >= 5)      # At least 5 events
]

# ======================================================================
# Code Block 6
# ======================================================================

from sklearn.ensemble import RandomForestRegressor

# Train on historical GPS sequences
# Features: current lat/lon, velocity, heading, time since permit
# Target: lat/lon 4 hours from now

# Flag if predicted trajectory intersects ROW buffer (50m)

# ======================================================================
# Code Block 7
# ======================================================================

# Create catalog and schema
spark.sql("CREATE CATALOG IF NOT EXISTS pipeline")
spark.sql("CREATE SCHEMA IF NOT EXISTS pipeline.row_monitoring")
spark.sql("USE pipeline.row_monitoring")

# Create Bronze tables (see data model above)
# Set up streaming ingestion with Auto Loader

# ======================================================================
# Code Block 8
# ======================================================================

# Databricks Notebook: Third-Party Work Detection
# Prerequisites: Bronze tables created, ROW centerline loaded

# COMMAND ----------
# Configuration
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, expr, current_timestamp
from pyspark.sql.types import DoubleType
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

spark = SparkSession.builder.getOrCreate()

CATALOG = 'pipeline'
SCHEMA = 'row_monitoring'
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------
# Haversine distance UDF
@udf(DoubleType())
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

spark.udf.register('haversine', haversine_distance)

# COMMAND ----------
# Unified event table
spark.sql(f"""
CREATE OR REPLACE TABLE silver.row_events AS
SELECT
    ticket_id AS event_id,
    'ticket_811' AS event_type,
    lat, lon,
    start_date AS event_date,
    has_active_watch,
    current_timestamp() AS processing_time
FROM bronze.tickets_811
WHERE start_date >= current_date() - INTERVAL 60 DAYS

UNION ALL

SELECT
    permit_id AS event_id,
    'permit' AS event_type,
    lat, lon,
    issue_date AS event_date,
    has_active_watch,
    current_timestamp() AS processing_time
FROM bronze.construction_permits
WHERE issue_date >= current_date() - INTERVAL 90 DAYS

UNION ALL

SELECT
    device_id AS event_id,
    'gps_equipment' AS event_type,
    lat, lon,
    date(timestamp) AS event_date,
    false AS has_active_watch,
    current_timestamp() AS processing_time
FROM bronze.gps_equipment
WHERE timestamp >= current_timestamp() - INTERVAL 7 DAYS
""")

# COMMAND ----------
# Compute distance to ROW
spark.sql(f"""
CREATE OR REPLACE TABLE silver.row_events_with_distance AS
SELECT
    e.*,
    MIN(haversine(e.lat, e.lon, r.lat, r.lon)) AS dist_to_row_km,
    DATEDIFF(CURRENT_DATE(), e.event_date) AS days_since_event
FROM silver.row_events e
CROSS JOIN reference.row_centerline r
GROUP BY e.event_id, e.event_type, e.lat, e.lon, e.event_date, 
         e.has_active_watch, e.processing_time
""")

# COMMAND ----------
# Risk scoring
df = spark.table('silver.row_events_with_distance').toPandas()

# Build components
components = pd.DataFrame({
    'distance_km': df['dist_to_row_km'],
    'days_old': df['days_since_event'],
    'no_watch': (~df['has_active_watch']).astype(float),
    'is_machinery': (df['event_type'] == 'gps_equipment').astype(float)
})

# Normalize
scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(components), columns=components.columns)

# Compute risk score
risk_score = (
    1.0 * (1 - normalized['distance_km']) +
    1.0 * (1 - normalized['days_old']) +
    0.8 * normalized['no_watch'] +
    0.7 * normalized['is_machinery']
)

df['risk_score'] = risk_score
df['alert_sent'] = False
df['alert_timestamp'] = None

# Save to Gold
spark.createDataFrame(df).write.mode('overwrite').saveAsTable('gold.row_risk_scores')
# print(f' Scored {len(df):,} events')

# COMMAND ----------
# Generate daily review list
spark.sql(f"""
CREATE OR REPLACE TABLE gold.daily_review_list AS
SELECT
    DATE(processing_time) AS review_date,
    event_id,
    event_type,
    lat,
    lon,
    dist_to_row_km,
    days_since_event,
    has_active_watch,
    risk_score,
    ROW_NUMBER() OVER (PARTITION BY DATE(processing_time) ORDER BY risk_score DESC) AS rank
FROM gold.row_risk_scores
QUALIFY rank <= 40
""")

# print(' Daily review list generated')

# COMMAND ----------
# Check for high-risk alerts
high_risk = spark.sql("""
SELECT * FROM gold.row_risk_scores
WHERE risk_score >= 0.8 AND alert_sent = FALSE
ORDER BY risk_score DESC
""")

if high_risk.count() > 0:
    # print(f'⚠️  {high_risk.count()} HIGH-RISK EVENTS require immediate action!')
    high_risk.show(10, truncate=False)
else:
    # print(' No high-risk events detected')

# ======================================================================
# Code Block 9
# ======================================================================

    pass
"""
Compute great-circle distance between two points in kilometers.
"""
R = 6371.0  # Earth radius in km

lat1_rad = np.radians(lat1)
lat2_rad = np.radians(lat2)
dlat = np.radians(lat2 - lat1)
dlon = np.radians(lon2 - lon1)

a = (np.sin(dlat / 2)**2 + 
     np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2)
c = 2 * np.arcsin(np.sqrt(a))

return R * c

# ======================================================================
# Code Block 10
# ======================================================================

# 'distance_km': df['dist_to_row_km'],
# 'days_old': df['days_since_event'],
# 'no_watch': (~df['has_active_watch']).astype(float),
# 'is_machinery': (df['event_type'] == 'gps_equipment').astype(float)

# ======================================================================
# Code Block 11
# ======================================================================

scaler.fit_transform(components),
columns=components.columns,
index=components.index

# ======================================================================
# Code Block 12
# ======================================================================

# 1.0 * (1 - normalized['distance_km']) +    # Closer = higher risk (40%)
# 1.0 * (1 - normalized['days_old']) +       # More recent = higher risk (30%)
# 0.8 * normalized['no_watch'] +             # No watch = higher risk (20%)
0.7 * normalized['is_machinery']           # Active machinery = higher risk (10%)

# ======================================================================
# Code Block 13
# ======================================================================

# AND alert_sent = FALSE

# ======================================================================
# Code Block 14
# ======================================================================

for row in high_risk_events.collect():
    # send_alert(
        recipient='patrol_team@pipeline.com',
        subject=f'CRITICAL: Third-party work detected at {row.lat:.5f}, {row.lon:.5f}',
        # body=f'''

# ======================================================================
# Code Block 15
# ======================================================================

spark.sql(f"""
UPDATE gold.row_risk_scores
SET alert_sent = TRUE, alert_timestamp = CURRENT_TIMESTAMP()
WHERE risk_score >= 0.8 AND alert_sent = FALSE
""")

# ======================================================================
# Code Block 16
# ======================================================================

CASE
# WHEN risk_score >= 0.8 THEN 'Critical (0.8)'
# WHEN risk_score >= 0.6 THEN 'High (0.6-0.8)'
# WHEN risk_score >= 0.4 THEN 'Moderate (0.4-0.6)'
# ELSE 'Low (<0.4)'
# END AS risk_category,
# COUNT(*) AS event_count

# ======================================================================
# Code Block 17
# ======================================================================

# - Third-party damages: 6/year → **1/year** (83% reduction)
# - Prevented incidents: 5  \$2.8M = **\$14M/year savings**

# ======================================================================
# Code Block 18
# ======================================================================

# (cluster_summary['event_type'] >= 2) &  # Multiple event types
(cluster_summary['event_id'] >= 5)      # At least 5 events

# ======================================================================
# Code Block 19
# ======================================================================

R = 6371.0
lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
dlat = np.radians(lat2 - lat1)
dlon = np.radians(lon2 - lon1)
a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
return 2 * R * np.arcsin(np.sqrt(a))

# ======================================================================
# Code Block 20
# ======================================================================

# 'distance_km': df['dist_to_row_km'],
# 'days_old': df['days_since_event'],
# 'no_watch': (~df['has_active_watch']).astype(float),
# 'is_machinery': (df['event_type'] == 'gps_equipment').astype(float)

# ======================================================================
# Code Block 21
# ======================================================================

# print(f'⚠️  {high_risk.count()} HIGH-RISK EVENTS require immediate action!')
high_risk.show(10, truncate=False)

# ======================================================================
# Code Block 22
# ======================================================================

# print(' No high-risk events detected')
