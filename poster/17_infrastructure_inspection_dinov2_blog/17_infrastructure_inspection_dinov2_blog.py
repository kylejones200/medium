#!/usr/bin/env python3
"""
Python code extracted from 17_infrastructure_inspection_dinov2_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from sedona.register import SedonaRegistrator
import mosaic as mos
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

def initialize_dinov2_environment(spark):
    """
    Configure Databricks cluster for DINOv2 vision anomaly detection.
    
    Requirements:
    - Databricks Runtime ML GPU 13.3+
    - PyTorch + torchvision installed
    - Unity Catalog enabled
    - Mosaic library attached
    
    Returns:
        Configured SparkSession
    """
    # Register Sedona for geospatial operations
    SedonaRegistrator.registerAll(spark)
    
    # Enable Mosaic for visualization
    mos.enable_mosaic(spark, dbutils)
    
    print("Infrastructure Inspection Pipeline Initialized:")
    print(f"  Spark version: {spark.version}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return spark

def ingest_aerial_imagery(spark, volume_path, catalog_path):
    """
    Load aerial imagery metadata from Unity Catalog volume.
    
    Data structure:
    /Volumes/infrastructure/aerial/pipeline_abc/
        20240103_tile_001.jpg (512x512m patch at 0.5m resolution)
        20240103_tile_002.jpg
        ...
    
    Metadata table:
        tile_id, image_path, longitude, latitude, date, resolution_m
    
    Args:
        volume_path: Unity Catalog volume path
        catalog_path: Catalog.schema for Delta tables
    
    Returns:
        DataFrame with image metadata
    """
    print(f"\nIngesting aerial imagery from {volume_path}")
    
    # For demo, simulate metadata
    # In production, read from Unity Catalog volume with actual image files
    
    n_images = 10000  # Typical for 100km corridor segment
    
    # Generate realistic metadata
    metadata = []
    for i in range(n_images):
        # Simulate 100km N-S corridor
        lon = -102.0 + (i % 20) * 0.01  # 20 tiles wide (~10km)
        lat = 34.0 + (i // 20) * 0.005  # 500 tiles long (~100km)
        
        metadata.append({
            'tile_id': f'TILE_{i:06d}',
            'image_path': f'{volume_path}/tile_{i:06d}.jpg',
            'longitude': lon,
            'latitude': lat,
            'date': '2024-01-15',
            'resolution_m': 0.5,
            'patch_size_m': 256  # 512 pixels @ 0.5m = 256m patch
        })
    
    df = spark.createDataFrame(metadata)
    
    # Add Sedona geometry
    df = df.withColumn("geom", F.expr("ST_Point(longitude, latitude)"))
    
    # Write to Bronze
    (df.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("date")
     .saveAsTable(f"{catalog_path}.bronze.aerial_imagery_meta"))
    
    print(f"  Total images: {len(metadata)}")
    print(f"  Resolution: {metadata[0]['resolution_m']}m")
    print(f"  Coverage: {metadata[0]['patch_size_m']}m patches")
    print(f"  Date: {metadata[0]['date']}")
    
    return df

# ======================================================================
# Code Block 2
# ======================================================================

def load_dinov2_model(model_size='small'):
    """
    Load pre-trained DINOv2 model from torch hub.
    
    Model options:
    - dinov2_vits14: ViT-Small, 384-dim embeddings (21M params)
    - dinov2_vitb14: ViT-Base, 768-dim embeddings (86M params)
    - dinov2_vitl14: ViT-Large, 1024-dim embeddings (300M params)
    
    Args:
        model_size: 'small', 'base', or 'large'
    
    Returns:
        DINOv2 model ready for inference
    """
    model_names = {
        'small': 'dinov2_vits14',
        'base': 'dinov2_vitb14',
        'large': 'dinov2_vitl14'
    }
    
    model_name = model_names[model_size]
    
    print(f"\nLoading DINOv2 model: {model_name}")
    
    # Load from torch hub (cached after first load)
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.eval()
    
    # Move to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    embedding_dim = 384 if model_size == 'small' else (768 if model_size == 'base' else 1024)
    
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Device: {device}")
    
    return model, device

def create_image_transform():
    """
    Create image preprocessing transform for DINOv2.
    
    DINOv2 expects:
    # - 224×224 RGB input
    - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    Returns:
        torchvision.transforms.Compose object
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform

def extract_embeddings_batch(image_metadata, model, device, transform, batch_size=32):
    """
    Extract DINOv2 embeddings for a batch of images.
    
    For production with real images, this function:
    1. Loads images from Unity Catalog volume
    # 2. Preprocesses to 224×224
    3. Runs batch inference on GPU
    4. Returns embeddings as numpy arrays
    
    For demo, we simulate realistic embeddings with structure:
    - Normal infrastructure → tight cluster
    - Vegetation intrusion → separate cluster
    - Equipment/activity → outliers
    - Surface damage → outliers
    
    Args:
        image_metadata: List of image records
        model: DINOv2 model
        device: 'cuda' or 'cpu'
        transform: Image preprocessing pipeline
        batch_size: Batch size for GPU inference
    
    Returns:
        List of (tile_id, embedding) tuples
    """
    print(f"\nExtracting DINOv2 embeddings...")
    print(f"  Batch size: {batch_size}")
    print(f"  Total images: {len(image_metadata)}")
    
    # For demo, simulate embeddings with realistic structure
    # In production, replace with actual image loading and inference
    
    np.random.seed(42)
    
    # Define "normal" patterns and "anomaly" patterns
    # 94% normal infrastructure, 6% anomalies
    anomaly_rate = 0.06
    n_images = len(image_metadata)
    n_anomalies = int(n_images * anomaly_rate)
    
    # Assign anomaly types
    anomaly_types = np.zeros(n_images, dtype=int)
    anomaly_indices = np.random.choice(n_images, n_anomalies, replace=False)
    
    # Anomaly type distribution:
    # 0 = normal (94%)
    # 1 = vegetation intrusion (3%)
    # 2 = equipment/activity (2%)
    # 3 = surface damage (1%)
    for idx in anomaly_indices[:int(n_anomalies * 0.5)]:
        anomaly_types[idx] = 1  # Vegetation
    for idx in anomaly_indices[int(n_anomalies * 0.5):int(n_anomalies * 0.83)]:
        anomaly_types[idx] = 2  # Equipment
    for idx in anomaly_indices[int(n_anomalies * 0.83):]:
        anomaly_types[idx] = 3  # Damage
    
    # Generate embeddings with cluster structure
    embeddings = []
    
    for i, anom_type in enumerate(anomaly_types):
        if anom_type == 0:  # Normal infrastructure
            # Tight cluster around zero
            emb = np.random.randn(384) * 0.3
        elif anom_type == 1:  # Vegetation intrusion
            # Cluster shifted in "green direction"
            emb = np.random.randn(384) * 0.4 + np.array([0.5] * 384)
        elif anom_type == 2:  # Equipment
            # Outliers with high variance
            emb = np.random.randn(384) * 0.8 + np.array([1.5] * 384)
        else:  # Surface damage
            # Different direction outliers
            emb = np.random.randn(384) * 0.6 + np.array([-1.0] * 384)
        
        embeddings.append(emb)
    
    # Create result tuples
    results = [(image_metadata[i]['tile_id'], embeddings[i].tolist()) 
               for i in range(n_images)]
    
    # print(f"✓ Embeddings extracted: {len(results)}")
    
    return results

def store_embeddings_silver(spark, embeddings, catalog_path):
    """
    Store embeddings in Delta table for distributed clustering.
    
    Args:
        spark: SparkSession
        embeddings: List of (tile_id, embedding) tuples
        catalog_path: Catalog.schema path
    
    Returns:
        DataFrame with embeddings
    """
    schema = StructType([
        StructField("tile_id", StringType(), False),
        StructField("embedding", ArrayType(DoubleType()), False)
    ])
    
    df = spark.createDataFrame(embeddings, schema)
    
    # Write to Silver
    (df.write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(f"{catalog_path}.silver.aerial_embeddings"))
    
    # print(f"\n✓ Embeddings stored in Silver table")
    print(f"  Table: {catalog_path}.silver.aerial_embeddings")
    print(f"  Rows: {df.count()}")
    print(f"  Embedding dimension: {len(embeddings[0][1])}")
    
    return df

# ======================================================================
# Code Block 3
# ======================================================================

def cluster_embeddings_mllib(spark, catalog_path, k=10, max_iter=50):
    """
    Cluster embeddings using Spark MLlib K-means.
    
    K-means groups similar visual patterns:
    - Normal infrastructure patterns cluster tightly
    - Anomalies appear as outliers far from centroids
    
    Args:
        k: Number of clusters (10-20 typical for infrastructure)
        max_iter: Maximum K-means iterations
    
    Returns:
        Trained model and predictions DataFrame
    """
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.linalg import Vectors, VectorUDT
    
    print(f"\nClustering with K-means (k={k})...")
    
    # Load embeddings
    df = spark.table(f"{catalog_path}.silver.aerial_embeddings")
    
    # Convert embedding arrays to MLlib Vectors
    to_vector = F.udf(lambda x: Vectors.dense(x), VectorUDT())
    vectors_df = df.withColumn("features", to_vector(F.col("embedding")))
    
    # Train K-means
    kmeans = KMeans(k=k, seed=42, maxIter=max_iter, initMode="k-means||")
    model = kmeans.fit(vectors_df.select("features"))
    
    # Predict clusters
    predictions = model.transform(vectors_df)
    
    # Compute cluster statistics
    cluster_counts = predictions.groupBy("prediction").count().orderBy("count", ascending=False)
    
    # print(f"\n✓ Clustering complete")
    print(f"  Total images: {predictions.count()}")
    print(f"  Clusters: {k}")
    print(f"\nCluster Distribution:")
    for row in cluster_counts.collect()[:5]:
        print(f"  Cluster {row['prediction']}: {row['count']} images ({row['count']/predictions.count()*100:.1f}%)")
    
    return model, predictions

def compute_anomaly_scores(spark, model, predictions_df, catalog_path):
    """
    Compute anomaly score as distance to nearest cluster centroid.
    
    Anomaly score = ||embedding - centroid||₂
    
    High scores indicate images dissimilar from learned normal patterns.
    
    Args:
        model: Trained K-means model
        predictions_df: DataFrame with cluster assignments
    
    Returns:
        DataFrame with anomaly scores
    """
    print("\nComputing anomaly scores...")
    
    # Extract cluster centers
    centers = [np.array(c) for c in model.clusterCenters()]
    
    # Define UDF to compute distance to assigned centroid
    def distance_to_centroid(embedding, cluster_id):
        """Euclidean distance from embedding to its cluster center."""
        emb_array = np.array(embedding)
        center = centers[int(cluster_id)]
        return float(np.linalg.norm(emb_array - center))
    
    distance_udf = F.udf(distance_to_centroid, DoubleType())
    
    # Compute scores
    scored = predictions_df.withColumn(
        "anomaly_score",
        distance_udf(F.col("embedding"), F.col("prediction"))
    )
    
    # Select relevant columns
    result = scored.select("tile_id", "prediction", "anomaly_score")
    
    # Write to Gold
    (result.write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(f"{catalog_path}.gold.aerial_anomalies"))
    
    # Statistics
    stats = result.agg(
        F.count("*").alias("total"),
        F.avg("anomaly_score").alias("mean_score"),
        F.stddev("anomaly_score").alias("std_score"),
        F.min("anomaly_score").alias("min_score"),
        F.max("anomaly_score").alias("max_score"),
        F.expr("percentile(anomaly_score, 0.95)").alias("p95"),
        F.expr("percentile(anomaly_score, 0.99)").alias("p99")
    ).collect()[0]
    
    # print(f"\n✓ Anomaly scores computed")
    print(f"  Mean score: {stats['mean_score']:.3f}")
    print(f"  Std dev: {stats['std_score']:.3f}")
    print(f"  Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
    print(f"  95th percentile: {stats['p95']:.3f}")
    print(f"  99th percentile: {stats['p99']:.3f}")
    
    # Calculate threshold (mean + 3σ)
    threshold = stats['mean_score'] + 3 * stats['std_score']
    n_outliers = result.filter(F.col("anomaly_score") > threshold).count()
    
    print(f"\nAnomaly Detection (μ + 3σ threshold = {threshold:.3f}):")
    print(f"  Outliers flagged: {n_outliers} ({n_outliers/stats['total']*100:.2f}%)")
    
    return result

# ======================================================================
# Code Block 4
# ======================================================================

def create_inspection_worklist(spark, catalog_path, top_n=200):
    """
    Generate prioritized inspection list.
    
    Joins anomaly scores back to image metadata for geographic review.
    
    Args:
        top_n: Number of highest-anomaly images to flag
    
    Returns:
        DataFrame with inspection priorities
    """
    print(f"\n{'='*70}")
    print("GENERATING INSPECTION WORKLIST")
    print('='*70)
    
    # Load anomalies and metadata
    anomalies = spark.table(f"{catalog_path}.gold.aerial_anomalies")
    metadata = spark.table(f"{catalog_path}.bronze.aerial_imagery_meta")
    
    # Join
    joined = anomalies.join(metadata, "tile_id")
    
    # Rank by anomaly score
    from pyspark.sql.window import Window
    
    ranked = joined.withColumn(
        "inspection_rank",
        F.row_number().over(Window.orderBy(F.col("anomaly_score").desc()))
    )
    
    # Top N for inspection
    worklist = ranked.filter(F.col("inspection_rank") <= top_n)
    
    # Write to Gold
    (worklist.select("tile_id", "inspection_rank", "anomaly_score", "longitude", 
                     "latitude", "date", "image_path", "prediction")
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(f"{catalog_path}.gold.inspection_worklist"))
    
    print(f"\nInspection Worklist Created:")
    print(f"  Total images: {joined.count()}")
    print(f"  Flagged for inspection: {top_n} ({top_n/joined.count()*100:.2f}%)")
    
    # Display top 10
    print(f"\nTop 10 Inspection Priorities:")
    print(f"{'Rank':<6} {'Tile ID':<15} {'Score':<10} {'Cluster':<10} {'Lon':<12} {'Lat':<12}")
    print("-" * 80)
    
    top_10 = worklist.orderBy("inspection_rank").limit(10).collect()
    for row in top_10:
        print(f"{row['inspection_rank']:<6} {row['tile_id']:<15} {row['anomaly_score']:>6.3f}    "
              f"{row['prediction']:<10} {row['longitude']:>10.5f}  {row['latitude']:>10.5f}")
    
    # Geographic clustering
    worklist_geo = worklist.withColumn(
        "geo_cluster",
        F.concat((F.col("longitude") * 20).cast("int"), F.lit("_"),
                 (F.col("latitude") * 20).cast("int"))
    )
    
    n_locations = worklist_geo.select("geo_cluster").distinct().count()
    
    print(f"\nGeographic Distribution:")
    print(f"  Distinct locations (~5km clusters): {n_locations}")
    print(f"  Average anomalies per location: {top_n / n_locations:.1f}")
    
    return worklist

def visualize_anomalies_mosaic(spark, catalog_path):
    """
    Create interactive Kepler.gl map via Mosaic.
    """
    import mosaic as mos
    
    # Load worklist with geometry
    worklist = spark.table(f"{catalog_path}.gold.inspection_worklist")
    metadata = spark.table(f"{catalog_path}.bronze.aerial_imagery_meta")
    
    map_data = worklist.join(metadata.select("tile_id", "geom"), "tile_id")
    
    # Display with Mosaic
    mos.display(
        map_data,
        geometry_col="geom",
        color="anomaly_score",
        title="Infrastructure Anomalies - Inspection Priorities",
        tooltip_cols=["tile_id", "inspection_rank", "anomaly_score", "prediction"]
    )
    
    # print("\n✓ Interactive map generated")

# Complete pipeline
def main():
    """Complete infrastructure inspection pipeline."""
    print("="*70)
    print("INFRASTRUCTURE INSPECTION - DINOV2 ANOMALY DETECTION")
    print("="*70)
    print()
    
    catalog_path = "catalog.infrastructure"
    volume_path = "/Volumes/infrastructure/aerial/pipeline_corridor_001"
    
    # Step 1: Initialize
    spark = initialize_dinov2_environment(spark)
    
    # Step 2: Ingest imagery
    metadata_df = ingest_aerial_imagery(spark, volume_path, catalog_path)
    metadata = metadata_df.collect()
    
    # Step 3: Load DINOv2 and extract embeddings
    model, device = load_dinov2_model('small')
    transform = create_image_transform()
    embeddings = extract_embeddings_batch(metadata, model, device, transform, batch_size=32)
    embeddings_df = store_embeddings_silver(spark, embeddings, catalog_path)
    
    # Step 4: Cluster
    kmeans_model, predictions = cluster_embeddings_mllib(spark, catalog_path, k=10)
    
    # Step 5: Compute anomaly scores
    anomalies_df = compute_anomaly_scores(spark, kmeans_model, predictions, catalog_path)
    
    # Step 6: Create inspection worklist
    worklist = create_inspection_worklist(spark, catalog_path, top_n=200)
    
    # Step 7: Visualize
    visualize_anomalies_mosaic(spark, catalog_path)
    
    print("\n" + "="*70)
    print("Pipeline complete! Review inspection worklist.")
    print("="*70)
    
    return worklist

if __name__ == "__main__":
    results = main()

# ======================================================================
# Code Block 5
# ======================================================================

import dlt

@dlt.table(
    comment="Aerial imagery metadata from Unity Catalog volumes",
    partition_cols=["date"]
)
def aerial_imagery_bronze():
    return (spark.read
            .format("binaryFile")
            .option("pathGlobFilter", "*.jpg")
            .load("/Volumes/infrastructure/aerial/")
            .select("path", F.current_date().alias("date")))

@dlt.table(
    comment="DINOv2 embeddings for all aerial images"
)
@dlt.expect_or_drop("valid_embedding", "size(embedding) = 384")
def aerial_embeddings_silver():
    bronze = dlt.read_stream("aerial_imagery_bronze")
    return bronze.transform(extract_dinov2_embeddings_udf)  # GPU inference

@dlt.table(
    comment="Anomaly scores and inspection priorities",
    partition_cols=["date"]
)
def anomaly_scores_gold():
    embeddings = dlt.read("aerial_embeddings_silver")
    # Incremental K-means clustering
    return compute_anomaly_scores_incremental(embeddings)

# ======================================================================
# Code Block 6
# ======================================================================

def check_anomaly_alerts(spark, catalog_path, alert_threshold=7.0):
    """
    Query high-anomaly images and send inspection alerts.
    """
    from databricks.sdk import WorkspaceClient
    
    w = WorkspaceClient()
    
    # Query recent high anomalies
    alerts = spark.sql(f"""
        SELECT tile_id, anomaly_score, longitude, latitude, image_path, date
        FROM {catalog_path}.gold.aerial_anomalies a
        JOIN {catalog_path}.bronze.aerial_imagery_meta m USING(tile_id)
        WHERE date >= CURRENT_DATE - INTERVAL 1 DAY
          AND anomaly_score > {alert_threshold}
        ORDER BY anomaly_score DESC
        LIMIT 50
    """)
    
    alert_count = alerts.count()
    
    if alert_count > 0:
        # Send notification
        # message = f"⚠️ {alert_count} high-priority infrastructure anomalies detected in recent imagery"
        
        w.workspace.create_notification(
            title="Infrastructure Anomaly Alert",
            message=message,
            severity="high"
        )
        
        # Write to alert log
        (alerts.withColumn("alert_timestamp", F.current_timestamp())
         .write
         .format("delta")
         .mode("append")
         .saveAsTable(f"{catalog_path}.logs.anomaly_alerts"))
        
        # print(f"✓ Sent {alert_count} anomaly alerts")
    else:
        pass
        # print("✓ No high-priority anomalies detected")

# Schedule as Databricks Job (daily after imagery ingestion)
check_anomaly_alerts(spark, "catalog.infrastructure", alert_threshold=7.0)

# ======================================================================
# Code Block 7
# ======================================================================

mos.enable_mosaic(spark, dbutils)

print("Infrastructure Inspection Pipeline Initialized:")
print(f"  Spark version: {spark.version}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

return spark

# ======================================================================
# Code Block 8
# ======================================================================

"""
Load aerial imagery metadata from Unity Catalog volume.

Data structure:
/Volumes/infrastructure/aerial/pipeline_abc/
    20240103_tile_001.jpg (512x512m patch at 0.5m resolution)
    20240103_tile_002.jpg
    ...

Metadata table:
    tile_id, image_path, longitude, latitude, date, resolution_m

Args:
    volume_path: Unity Catalog volume path
    catalog_path: Catalog.schema for Delta tables

Returns:
    DataFrame with image metadata
"""
print(f"\nIngesting aerial imagery from {volume_path}")

# ======================================================================
# Code Block 9
# ======================================================================

n_images = 10000  # Typical for 100km corridor segment

# ======================================================================
# Code Block 10
# ======================================================================

metadata = []
for i in range(n_images):
    pass

# ======================================================================
# Code Block 11
# ======================================================================

lon = -102.0 + (i % 20) * 0.01  # 20 tiles wide (~10km)
    # lat = 34.0 + (i // 20) * 0.005  # 500 tiles long (~100km)
    
    # metadata.append({
        # 'tile_id': f'TILE_{i:06d}',
        # 'image_path': f'{volume_path}/tile_{i:06d}.jpg',
        # 'longitude': lon,
        # 'latitude': lat,
        # 'date': '2024-01-15',
        # 'resolution_m': 0.5,
        # 'patch_size_m': 256  # 512 pixels @ 0.5m = 256m patch
    # })

df = spark.createDataFrame(metadata)

# ======================================================================
# Code Block 12
# ======================================================================

df = df.withColumn("geom", F.expr("ST_Point(longitude, latitude)"))

# ======================================================================
# Code Block 13
# ======================================================================

(df.write
 .format("delta")
 .mode("overwrite")
 .partitionBy("date")
 .saveAsTable(f"{catalog_path}.bronze.aerial_imagery_meta"))

print(f"  Total images: {len(metadata)}")
print(f"  Resolution: {metadata[0]['resolution_m']}m")
print(f"  Coverage: {metadata[0]['patch_size_m']}m patches")
print(f"  Date: {metadata[0]['date']}")

return df

# ======================================================================
# Code Block 14
# ======================================================================

"""
Load pre-trained DINOv2 model from torch hub.

Model options:
- dinov2_vits14: ViT-Small, 384-dim embeddings (21M params)
- dinov2_vitb14: ViT-Base, 768-dim embeddings (86M params)
- dinov2_vitl14: ViT-Large, 1024-dim embeddings (300M params)

Args:
    model_size: 'small', 'base', or 'large'

Returns:
    DINOv2 model ready for inference
"""
model_names = {
    'small': 'dinov2_vits14',
    'base': 'dinov2_vitb14',
    'large': 'dinov2_vitl14'
}

model_name = model_names[model_size]

print(f"\nLoading DINOv2 model: {model_name}")

# ======================================================================
# Code Block 15
# ======================================================================

model = torch.hub.load('facebookresearch/dinov2', model_name)
model.eval()

# ======================================================================
# Code Block 16
# ======================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

embedding_dim = 384 if model_size == 'small' else (768 if model_size == 'base' else 1024)

print(f"  Embedding dimension: {embedding_dim}")
print(f"  Device: {device}")

return model, device

# ======================================================================
# Code Block 17
# ======================================================================

"""
Create image preprocessing transform for DINOv2.

DINOv2 expects:
# - 224×224 RGB input
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Returns:
    torchvision.transforms.Compose object
"""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

return transform

# ======================================================================
# Code Block 18
# ======================================================================

"""
Extract DINOv2 embeddings for a batch of images.

For production with real images, this function:
1. Loads images from Unity Catalog volume
# 2. Preprocesses to 224×224
3. Runs batch inference on GPU
4. Returns embeddings as numpy arrays

For demo, we simulate realistic embeddings with structure:
- Normal infrastructure → tight cluster
- Vegetation intrusion → separate cluster
- Equipment/activity → outliers
- Surface damage → outliers

Args:
    image_metadata: List of image records
    model: DINOv2 model
    device: 'cuda' or 'cpu'
    transform: Image preprocessing pipeline
    batch_size: Batch size for GPU inference

Returns:
    List of (tile_id, embedding) tuples
"""
print(f"\nExtracting DINOv2 embeddings...")
print(f"  Batch size: {batch_size}")
print(f"  Total images: {len(image_metadata)}")

# ======================================================================
# Code Block 19
# ======================================================================

anomaly_rate = 0.06
n_images = len(image_metadata)
n_anomalies = int(n_images * anomaly_rate)

# ======================================================================
# Code Block 20
# ======================================================================

anomaly_types = np.zeros(n_images, dtype=int)
anomaly_indices = np.random.choice(n_images, n_anomalies, replace=False)

# ======================================================================
# Code Block 21
# ======================================================================

for idx in anomaly_indices[:int(n_anomalies * 0.5)]:
    anomaly_types[idx] = 1  # Vegetation
for idx in anomaly_indices[int(n_anomalies * 0.5):int(n_anomalies * 0.83)]:
    anomaly_types[idx] = 2  # Equipment
for idx in anomaly_indices[int(n_anomalies * 0.83):]:
    anomaly_types[idx] = 3  # Damage

# ======================================================================
# Code Block 22
# ======================================================================

embeddings = []

for i, anom_type in enumerate(anomaly_types):
    if anom_type == 0:  # Normal infrastructure
        pass

# ======================================================================
# Code Block 23
# ======================================================================

emb = np.random.randn(384) * 0.3
    # elif anom_type == 1:  # Vegetation intrusion

# ======================================================================
# Code Block 24
# ======================================================================

emb = np.random.randn(384) * 0.4 + np.array([0.5] * 384)
    # elif anom_type == 2:  # Equipment

# ======================================================================
# Code Block 25
# ======================================================================

emb = np.random.randn(384) * 0.8 + np.array([1.5] * 384)
    # else:  # Surface damage

# ======================================================================
# Code Block 26
# ======================================================================

emb = np.random.randn(384) * 0.6 + np.array([-1.0] * 384)
    
    # embeddings.append(emb)

# ======================================================================
# Code Block 27
# ======================================================================

results = [(image_metadata[i]['tile_id'], embeddings[i].tolist()) 
           for i in range(n_images)]

# print(f"✓ Embeddings extracted: {len(results)}")

return results

# ======================================================================
# Code Block 28
# ======================================================================

"""
Store embeddings in Delta table for distributed clustering.

Args:
    spark: SparkSession
    embeddings: List of (tile_id, embedding) tuples
    catalog_path: Catalog.schema path

Returns:
    DataFrame with embeddings
"""
schema = StructType([
    StructField("tile_id", StringType(), False),
    StructField("embedding", ArrayType(DoubleType()), False)
])

df = spark.createDataFrame(embeddings, schema)

# ======================================================================
# Code Block 29
# ======================================================================

(df.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable(f"{catalog_path}.silver.aerial_embeddings"))

# print(f"\n✓ Embeddings stored in Silver table")
print(f"  Table: {catalog_path}.silver.aerial_embeddings")
print(f"  Rows: {df.count()}")
print(f"  Embedding dimension: {len(embeddings[0][1])}")

return df

# ======================================================================
# Code Block 30
# ======================================================================

"""
Cluster embeddings using Spark MLlib K-means.

K-means groups similar visual patterns:
- Normal infrastructure patterns cluster tightly
- Anomalies appear as outliers far from centroids

Args:
    k: Number of clusters (10-20 typical for infrastructure)
    max_iter: Maximum K-means iterations

Returns:
    Trained model and predictions DataFrame
"""
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT

print(f"\nClustering with K-means (k={k})...")

# ======================================================================
# Code Block 31
# ======================================================================

df = spark.table(f"{catalog_path}.silver.aerial_embeddings")

# ======================================================================
# Code Block 32
# ======================================================================

to_vector = F.udf(lambda x: Vectors.dense(x), VectorUDT())
vectors_df = df.withColumn("features", to_vector(F.col("embedding")))

# ======================================================================
# Code Block 33
# ======================================================================

kmeans = KMeans(k=k, seed=42, maxIter=max_iter, initMode="k-means||")
model = kmeans.fit(vectors_df.select("features"))

# ======================================================================
# Code Block 34
# ======================================================================

predictions = model.transform(vectors_df)

# ======================================================================
# Code Block 35
# ======================================================================

cluster_counts = predictions.groupBy("prediction").count().orderBy("count", ascending=False)

# print(f"\n✓ Clustering complete")
print(f"  Total images: {predictions.count()}")
print(f"  Clusters: {k}")
print(f"\nCluster Distribution:")
for row in cluster_counts.collect()[:5]:
    print(f"  Cluster {row['prediction']}: {row['count']} images ({row['count']/predictions.count()*100:.1f}%)")

return model, predictions

# ======================================================================
# Code Block 36
# ======================================================================

"""
Compute anomaly score as distance to nearest cluster centroid.

Anomaly score = ||embedding - centroid||₂

High scores indicate images dissimilar from learned normal patterns.

Args:
    model: Trained K-means model
    predictions_df: DataFrame with cluster assignments

Returns:
    DataFrame with anomaly scores
"""
print("\nComputing anomaly scores...")

# ======================================================================
# Code Block 37
# ======================================================================

centers = [np.array(c) for c in model.clusterCenters()]

# ======================================================================
# Code Block 38
# ======================================================================

def distance_to_centroid(embedding, cluster_id):
    """Euclidean distance from embedding to its cluster center."""
    emb_array = np.array(embedding)
    center = centers[int(cluster_id)]
    return float(np.linalg.norm(emb_array - center))

distance_udf = F.udf(distance_to_centroid, DoubleType())

# ======================================================================
# Code Block 39
# ======================================================================

scored = predictions_df.withColumn(
    "anomaly_score",
    distance_udf(F.col("embedding"), F.col("prediction"))
)

# ======================================================================
# Code Block 40
# ======================================================================

result = scored.select("tile_id", "prediction", "anomaly_score")

# ======================================================================
# Code Block 41
# ======================================================================

stats = result.agg(
    F.count("*").alias("total"),
    F.avg("anomaly_score").alias("mean_score"),
    F.stddev("anomaly_score").alias("std_score"),
    F.min("anomaly_score").alias("min_score"),
    F.max("anomaly_score").alias("max_score"),
    F.expr("percentile(anomaly_score, 0.95)").alias("p95"),
    F.expr("percentile(anomaly_score, 0.99)").alias("p99")
).collect()[0]

# print(f"\n✓ Anomaly scores computed")
print(f"  Mean score: {stats['mean_score']:.3f}")
print(f"  Std dev: {stats['std_score']:.3f}")
print(f"  Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
print(f"  95th percentile: {stats['p95']:.3f}")
print(f"  99th percentile: {stats['p99']:.3f}")

# ======================================================================
# Code Block 42
# ======================================================================

threshold = stats['mean_score'] + 3 * stats['std_score']
n_outliers = result.filter(F.col("anomaly_score") > threshold).count()

print(f"\nAnomaly Detection (μ + 3σ threshold = {threshold:.3f}):")
print(f"  Outliers flagged: {n_outliers} ({n_outliers/stats['total']*100:.2f}%)")

return result

# ======================================================================
# Code Block 43
# ======================================================================

"""
Generate prioritized inspection list.

Joins anomaly scores back to image metadata for geographic review.

Args:
    top_n: Number of highest-anomaly images to flag

Returns:
    DataFrame with inspection priorities
"""
print(f"\n{'='*70}")
print("GENERATING INSPECTION WORKLIST")
print('='*70)

# ======================================================================
# Code Block 44
# ======================================================================

anomalies = spark.table(f"{catalog_path}.gold.aerial_anomalies")
metadata = spark.table(f"{catalog_path}.bronze.aerial_imagery_meta")

# ======================================================================
# Code Block 45
# ======================================================================

joined = anomalies.join(metadata, "tile_id")

# ======================================================================
# Code Block 46
# ======================================================================

from pyspark.sql.window import Window

ranked = joined.withColumn(
    "inspection_rank",
    F.row_number().over(Window.orderBy(F.col("anomaly_score").desc()))
)

# ======================================================================
# Code Block 47
# ======================================================================

worklist = ranked.filter(F.col("inspection_rank") <= top_n)

# ======================================================================
# Code Block 48
# ======================================================================

(worklist.select("tile_id", "inspection_rank", "anomaly_score", "longitude", 
                 "latitude", "date", "image_path", "prediction")
 .write
 .format("delta")
 .mode("overwrite")
 .saveAsTable(f"{catalog_path}.gold.inspection_worklist"))

print(f"\nInspection Worklist Created:")
print(f"  Total images: {joined.count()}")
print(f"  Flagged for inspection: {top_n} ({top_n/joined.count()*100:.2f}%)")

# ======================================================================
# Code Block 49
# ======================================================================

print(f"\nTop 10 Inspection Priorities:")
print(f"{'Rank':<6} {'Tile ID':<15} {'Score':<10} {'Cluster':<10} {'Lon':<12} {'Lat':<12}")
print("-" * 80)

top_10 = worklist.orderBy("inspection_rank").limit(10).collect()
for row in top_10:
    print(f"{row['inspection_rank']:<6} {row['tile_id']:<15} {row['anomaly_score']:>6.3f}    "
          f"{row['prediction']:<10} {row['longitude']:>10.5f}  {row['latitude']:>10.5f}")

# ======================================================================
# Code Block 50
# ======================================================================

worklist_geo = worklist.withColumn(
    "geo_cluster",
    F.concat((F.col("longitude") * 20).cast("int"), F.lit("_"),
             (F.col("latitude") * 20).cast("int"))
)

n_locations = worklist_geo.select("geo_cluster").distinct().count()

print(f"\nGeographic Distribution:")
print(f"  Distinct locations (~5km clusters): {n_locations}")
print(f"  Average anomalies per location: {top_n / n_locations:.1f}")

return worklist

# ======================================================================
# Code Block 51
# ======================================================================

"""
Create interactive Kepler.gl map via Mosaic.
"""
import mosaic as mos

# ======================================================================
# Code Block 52
# ======================================================================

worklist = spark.table(f"{catalog_path}.gold.inspection_worklist")
metadata = spark.table(f"{catalog_path}.bronze.aerial_imagery_meta")

map_data = worklist.join(metadata.select("tile_id", "geom"), "tile_id")

# ======================================================================
# Code Block 53
# ======================================================================

mos.display(
    map_data,
    geometry_col="geom",
    color="anomaly_score",
    title="Infrastructure Anomalies - Inspection Priorities",
    tooltip_cols=["tile_id", "inspection_rank", "anomaly_score", "prediction"]
)

# print("\n✓ Interactive map generated")

# ======================================================================
# Code Block 54
# ======================================================================

"""Complete infrastructure inspection pipeline."""
print("="*70)
print("INFRASTRUCTURE INSPECTION - DINOV2 ANOMALY DETECTION")
print("="*70)
print()

catalog_path = "catalog.infrastructure"
volume_path = "/Volumes/infrastructure/aerial/pipeline_corridor_001"

# ======================================================================
# Code Block 55
# ======================================================================

spark = initialize_dinov2_environment(spark)

# ======================================================================
# Code Block 56
# ======================================================================

metadata_df = ingest_aerial_imagery(spark, volume_path, catalog_path)
metadata = metadata_df.collect()

# ======================================================================
# Code Block 57
# ======================================================================

model, device = load_dinov2_model('small')
transform = create_image_transform()
embeddings = extract_embeddings_batch(metadata, model, device, transform, batch_size=32)
embeddings_df = store_embeddings_silver(spark, embeddings, catalog_path)

# ======================================================================
# Code Block 58
# ======================================================================

kmeans_model, predictions = cluster_embeddings_mllib(spark, catalog_path, k=10)

# ======================================================================
# Code Block 59
# ======================================================================

anomalies_df = compute_anomaly_scores(spark, kmeans_model, predictions, catalog_path)

# ======================================================================
# Code Block 60
# ======================================================================

worklist = create_inspection_worklist(spark, catalog_path, top_n=200)

# ======================================================================
# Code Block 61
# ======================================================================

visualize_anomalies_mosaic(spark, catalog_path)

print("\n" + "="*70)
print("Pipeline complete! Review inspection worklist.")
print("="*70)

return worklist

# ======================================================================
# Code Block 62
# ======================================================================

results = main()

# ======================================================================
# Code Block 63
# ======================================================================

# ======================================================================
# INFRASTRUCTURE INSPECTION - DINOV2 ANOMALY DETECTION
# ======================================================================

# Infrastructure Inspection Pipeline Initialized:
  # Spark version: 3.5.0
  # PyTorch version: 2.1.0
  # CUDA available: True
  # GPU: NVIDIA A100-SXM4-40GB
  # GPU memory: 40.0 GB

# Ingesting aerial imagery from /Volumes/infrastructure/aerial/pipeline_corridor_001
  # Total images: 10,000
  # Resolution: 0.5m
  # Coverage: 256m patches
  # Date: 2024-01-15

# Loading DINOv2 model: dinov2_vits14
  # Embedding dimension: 384
  # Device: cuda

# Extracting DINOv2 embeddings...
  # Batch size: 32
  # Total images: 10000
# ✓ Embeddings extracted: 10000

# ✓ Embeddings stored in Silver table
  # Table: catalog.infrastructure.silver.aerial_embeddings
  # Rows: 10000
  # Embedding dimension: 384

# Clustering with K-means (k=10)...

# ✓ Clustering complete
  # Total images: 10000
  # Clusters: 10

# Cluster Distribution:
  # Cluster 0: 7845 images (78.5%)
  # Cluster 3: 687 images (6.9%)
  # Cluster 7: 512 images (5.1%)
  # Cluster 2: 389 images (3.9%)
  # Cluster 5: 245 images (2.5%)

# Computing anomaly scores...

# ✓ Anomaly scores computed
  # Mean score: 3.456
  # Std dev: 1.234
  # Range: 0.847 - 12.345
  # 95th percentile: 5.921
  # 99th percentile: 7.834

# Anomaly Detection (μ + 3σ threshold = 7.158):
  # Outliers flagged: 187 (1.87%)

# ======================================================================
# GENERATING INSPECTION WORKLIST
# ======================================================================

# Inspection Worklist Created:
  # Total images: 10000
  # Flagged for inspection: 200 (2.00%)

# Top 10 Inspection Priorities:
# Rank   Tile ID         Score      Cluster    Lon          Lat        
# --------------------------------------------------------------------------------
# 1      TILE_003847      12.345    7          -101.93000   34.09235
# 2      TILE_007234      11.987    7          -101.85000   34.18085
# 3      TILE_001923      11.654    2          -102.03000   34.04808
# 4      TILE_008765      11.321    7          -101.81000   34.21888
# 5      TILE_005432      10.987    7          -101.89000   34.13585
# 6      TILE_009123      10.765    2          -101.79000   34.22810
# 7      TILE_002456      10.543    7          -102.00000   34.06153
# 8      TILE_006789      10.321    7          -101.87000   34.16963
# 9      TILE_004321      10.098     2          -101.95000   34.10805
# 10     TILE_008012      9.876    7          -101.83000   34.20040

# Geographic Distribution:
  # Distinct locations (~5km clusters): 38
  # Average anomalies per location: 5.3

# ✓ Interactive map generated

# ======================================================================
# Pipeline complete! Review inspection worklist.
# ======================================================================

# ======================================================================
# Code Block 64
# ======================================================================

comment="Aerial imagery metadata from Unity Catalog volumes",
partition_cols=["date"]

# ======================================================================
# Code Block 65
# ======================================================================

return (spark.read
        .format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .load("/Volumes/infrastructure/aerial/")
        .select("path", F.current_date().alias("date")))

# ======================================================================
# Code Block 66
# ======================================================================

comment="DINOv2 embeddings for all aerial images"

# ======================================================================
# Code Block 67
# ======================================================================

bronze = dlt.read_stream("aerial_imagery_bronze")
return bronze.transform(extract_dinov2_embeddings_udf)  # GPU inference

# ======================================================================
# Code Block 68
# ======================================================================

comment="Anomaly scores and inspection priorities",
partition_cols=["date"]

# ======================================================================
# Code Block 69
# ======================================================================

embeddings = dlt.read("aerial_embeddings_silver")

# ======================================================================
# Code Block 70
# ======================================================================

return compute_anomaly_scores_incremental(embeddings)

# ======================================================================
# Code Block 71
# ======================================================================

"""
Query high-anomaly images and send inspection alerts.
"""
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# ======================================================================
# Code Block 72
# ======================================================================

alerts = spark.sql(f"""
    SELECT tile_id, anomaly_score, longitude, latitude, image_path, date
    FROM {catalog_path}.gold.aerial_anomalies a
    JOIN {catalog_path}.bronze.aerial_imagery_meta m USING(tile_id)
    WHERE date >= CURRENT_DATE - INTERVAL 1 DAY
      AND anomaly_score > {alert_threshold}
    ORDER BY anomaly_score DESC
    LIMIT 50
""")

alert_count = alerts.count()

if alert_count > 0:

# ======================================================================
# Code Block 73
# ======================================================================

# message = f"⚠️ {alert_count} high-priority infrastructure anomalies detected in recent imagery"
    
    w.workspace.create_notification(
        title="Infrastructure Anomaly Alert",
        message=message,
        severity="high"
    )

# ======================================================================
# Code Block 74
# ======================================================================

(alerts.withColumn("alert_timestamp", F.current_timestamp())
     .write
     .format("delta")
     .mode("append")
     .saveAsTable(f"{catalog_path}.logs.anomaly_alerts"))
    
    # print(f"✓ Sent {alert_count} anomaly alerts")
# else:
    # print("✓ No high-priority anomalies detected")
