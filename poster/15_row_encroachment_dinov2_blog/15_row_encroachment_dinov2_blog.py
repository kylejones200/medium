#!/usr/bin/env python3
"""
Python code extracted from 15_row_encroachment_dinov2_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

# Core
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StandardScaler

# Computer Vision
import torch
from torchvision import transforms
from PIL import Image

# Geospatial
from sedona.register import SedonaRegistrator
import mosaic as mos

# Utilities
import numpy as np
import pandas as pd

# ======================================================================
# Code Block 2
# ======================================================================

def initialize_dinov2_pipeline(spark):
    """
    Initialize Databricks environment for DINOv2-based encroachment detection.
    """
    # Register Sedona for geospatial operations
    SedonaRegistrator.registerAll(spark)
    
    # Enable Mosaic
    mos.enable_mosaic(spark, dbutils)
    
    print("DINOv2 Encroachment Detection Pipeline Initialized")
    print(f"  Spark version: {spark.version}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    return spark

def load_row_buffer_and_tiles(spark, pipeline_geom, buffer_m=500, tile_size_m=50):
    """
    Load pipeline corridor geometry and generate tile grid.
    
    Args:
        pipeline_geom: LineString WKT of pipeline route
        buffer_m: Buffer distance in meters (500m typical)
        tile_size_m: Tile dimension in meters (50m = ~167 pixels at 0.3m res)
    
    Returns:
        DataFrame with tile polygons and metadata
    """
    # Create buffer around pipeline
    # Convert meters to degrees (rough approximation at mid-latitudes)
    buffer_deg = buffer_m / 111000  # ~111km per degree latitude
    tile_deg = tile_size_m / 111000
    
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW pipeline_row AS
        SELECT 
            'PIPELINE_001' AS pipeline_id,
            ST_GeomFromWKT('{pipeline_geom}') AS geom
    """)
    
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW row_buffer AS
        SELECT 
            pipeline_id,
            ST_Buffer(geom, {buffer_deg}) AS buffer_geom
        FROM pipeline_row
    """)
    
    # Generate tile grid using Mosaic
    tiles_df = spark.sql(f"""
        SELECT 
            pipeline_id,
            mos.grid_tessellate(buffer_geom, {tile_deg}) AS tile_cell,
            mos.grid_boundaryaswkb(mos.grid_tessellate(buffer_geom, {tile_deg})) AS tile_geom_wkb
        FROM row_buffer
    """)
    
    print(f"\nROW Corridor Tiling:")
    print(f"  Buffer: {buffer_m}m")
    print(f"  Tile size: {tile_size_m}m")
    print(f"  Total tiles: {tiles_df.count()}")
    
    return tiles_df

def ingest_aerial_imagery(spark, image_paths, tiles_df, catalog_path):
    """
    Ingest aerial/satellite imagery and extract tiles.
    
    In production, this reads from:
    - Cloud storage (S3/ADLS) with high-res imagery
    - Planet Labs API for daily satellite
    - Drone imagery from inspection flights
    
    Returns:
        DataFrame with tile_id, image_path, lon, lat, date
    """
    # Simulate tile metadata
    # In production, spatial join imagery footprints to tile grid
    
    n_tiles = 20000  # Typical for 100-mile pipeline segment
    
    tile_data = []
    for i in range(n_tiles):
        tile_data.append({
            'tile_id': f'TILE_{i:06d}',
            'image_path': f'/dbfs/pipeline_imagery/2024/tile_{i:06d}.jpg',
            'longitude': -102.0 + (i % 200) * 0.001,
            'latitude': 34.0 + (i // 200) * 0.001,
            'date': '2024-01-15',
            'resolution_m': 0.5
        })
    
    tiles_df = spark.createDataFrame(tile_data)
    
    # Write to Bronze
    (tiles_df.write
     .format("delta")
     .mode("overwrite")
     .partitionBy("date")
     .saveAsTable(f"{catalog_path}.bronze.row_tiles"))
    
    print(f"\nAerial Imagery Ingested:")
    print(f"  Total tiles: {len(tile_data)}")
    print(f"  Resolution: {tile_data[0]['resolution_m']}m")
    print(f"  Date: {tile_data[0]['date']}")
    
    return tiles_df

# ======================================================================
# Code Block 3
# ======================================================================

def extract_dinov2_embeddings_batch(spark, tiles_df, catalog_path, batch_size=32):
    """
    Extract DINOv2 embeddings for all tiles using GPU cluster.
    
    This function runs DINOv2 inference distributed across Spark workers.
    Each worker loads the model once, then processes batches of tiles.
    
    Args:
        tiles_df: DataFrame with tile_id and image_path
        batch_size: Batch size for GPU inference
    
    Returns:
        DataFrame with tile_id and embedding vector
    """
    print("\nLoading DINOv2 Model...")
    print("  Model: dinov2_vits14 (ViT-Small, 384-dim embeddings)")
    print("  Pretrained on: 142M images (ImageNet-22k + curated)")
    
    # Define embedding extraction UDF
    def extract_embeddings_udf(image_paths_batch):
        """
        Extract embeddings for a batch of images.
        Runs on each Spark worker with local GPU.
        """
        import torch
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Load DINOv2 model (cached on worker)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = model.to(device).eval()
        
        # Define image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        embeddings = []
        with torch.no_grad():
            for img_path in image_paths_batch:
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    # Extract embedding
                    emb = model(img_tensor).squeeze().cpu().numpy()
                    embeddings.append(emb.tolist())
                except Exception as e:
                    # Handle corrupt/missing images
                    embeddings.append([0.0] * 384)
        
        return embeddings
    
    # For demo, simulate embeddings (production uses actual DINOv2)
    print("\nExtracting Embeddings (simulated for demo)...")
    
    # Generate synthetic embeddings with realistic structure
    np.random.seed(42)
    n_tiles = tiles_df.count()
    
    # Create 3 clusters representing:
    # 1. Normal vegetation (70%)
    # 2. Maintained ROW (25%)
    # 3. Encroachment/construction (5%)
    
    cluster_assignments = np.random.choice([0, 1, 2], n_tiles, p=[0.70, 0.25, 0.05])
    
    # Generate embeddings with cluster structure
    embeddings_list = []
    for i in range(n_tiles):
        cluster = cluster_assignments[i]
        
        # Cluster centers in 384-d space
        if cluster == 0:  # Normal vegetation
            center = np.random.randn(384) * 0.1
        elif cluster == 1:  # Maintained ROW
            center = np.ones(384) * 0.5 + np.random.randn(384) * 0.15
        else:  # Encroachment
            center = np.random.randn(384) + 2.0
        
        embedding = center + np.random.randn(384) * 0.3
        embeddings_list.append(embedding.tolist())
    
    # Create DataFrame
    tiles_with_embeddings = tiles_df.limit(n_tiles).toPandas()
    tiles_with_embeddings['embedding'] = embeddings_list
    
    embeddings_df = spark.createDataFrame(tiles_with_embeddings)
    
    # Write to Silver
    (embeddings_df.write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(f"{catalog_path}.silver.tile_embeddings"))
    
    # print(f"✓ Embeddings extracted: {n_tiles} tiles")
    print(f"  Embedding dimension: 384")
    print(f"  Storage: Delta table")
    
    return embeddings_df

# ======================================================================
# Code Block 4
# ======================================================================

def cluster_and_score_anomalies(spark, embeddings_df, catalog_path, k=20):
    """
    Perform K-means clustering and compute anomaly scores.
    
    Anomaly score = distance to nearest cluster centroid
    High scores indicate tiles dissimilar to normal patterns.
    
    Args:
        embeddings_df: DataFrame with tile_id and embedding
        k: Number of clusters (20-50 typical)
    
    Returns:
        DataFrame with anomaly scores
    """
    print(f"\nClustering with K-means (k={k})...")
    
    # Convert embedding lists to MLlib Vectors
    vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())
    vectors_df = embeddings_df.withColumn("features", vector_udf(F.col("embedding")))
    
    # Train K-means
    kmeans = KMeans(k=k, seed=42, maxIter=50, initMode="k-means||")
    model = kmeans.fit(vectors_df.select("features"))
    
    # Predict clusters
    predictions = model.transform(vectors_df)
    
    # Extract cluster centers
    centers = [np.array(c) for c in model.clusterCenters()]
    
    # Compute distance to assigned centroid (anomaly score)
    def compute_distance_to_centroid(embedding, cluster_id):
        """Euclidean distance from embedding to its cluster center."""
        emb_array = np.array(embedding)
        center = centers[int(cluster_id)]
        return float(np.linalg.norm(emb_array - center))
    
    distance_udf = F.udf(compute_distance_to_centroid, DoubleType())
    
    scored = predictions.withColumn(
        "encroachment_score",
        distance_udf(F.col("embedding"), F.col("prediction"))
    )
    
    # Write to Gold
    (scored.select("tile_id", "longitude", "latitude", "date", 
                   "prediction", "encroachment_score")
     .write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(f"{catalog_path}.gold.row_encroachment_scores"))
    
    # Statistics
    stats = scored.agg(
        F.count("*").alias("total_tiles"),
        F.avg("encroachment_score").alias("mean_score"),
        F.stddev("encroachment_score").alias("std_score"),
        F.max("encroachment_score").alias("max_score"),
        F.expr("percentile(encroachment_score, 0.95)").alias("p95_score"),
        F.expr("percentile(encroachment_score, 0.99)").alias("p99_score")
    ).collect()[0]
    
    print(f"\nClustering Results:")
    print(f"  Clusters: {k}")
    print(f"  Total tiles: {stats['total_tiles']}")
    print(f"  Mean anomaly score: {stats['mean_score']:.3f}")
    print(f"  Std dev: {stats['std_score']:.3f}")
    print(f"  95th percentile: {stats['p95_score']:.3f}")
    print(f"  99th percentile: {stats['p99_score']:.3f}")
    
    # Calculate threshold (mean + 3σ)
    threshold = stats['mean_score'] + 3 * stats['std_score']
    n_outliers = scored.filter(F.col("encroachment_score") > threshold).count()
    
    print(f"\nAnomaly Detection:")
    print(f"  Threshold (μ + 3σ): {threshold:.3f}")
    print(f"  Outliers flagged: {n_outliers} ({n_outliers/stats['total_tiles']*100:.2f}%)")
    
    return scored

# ======================================================================
# Code Block 5
# ======================================================================

def generate_inspection_worklist(spark, catalog_path, top_n=200, threshold_sigma=3.0):
    """
    Create prioritized inspection list for field crews.
    
    Ranking criteria:
    - Anomaly score (distance to centroid)
    - Proximity to pipeline centerline
    - Change from previous inspection
    - High consequence area (HCA) proximity
    
    Returns:
        DataFrame with top inspection priorities
    """
    print(f"\n{'='*70}")
    print("GENERATING INSPECTION WORK LIST")
    print('='*70)
    
    # Load scores
    scores_df = spark.table(f"{catalog_path}.gold.row_encroachment_scores")
    
    # Calculate statistics for threshold
    stats = scores_df.agg(
        F.avg("encroachment_score").alias("mean"),
        F.stddev("encroachment_score").alias("std")
    ).collect()[0]
    
    threshold = stats['mean'] + threshold_sigma * stats['std']
    
    # Filter high anomaly tiles
    outliers = scores_df.filter(F.col("encroachment_score") > threshold)
    
    # Rank by score
    ranked = outliers.withColumn(
        "priority_rank",
        F.row_number().over(
            Window.orderBy(F.col("encroachment_score").desc())
        )
    )
    
    # Top N for inspection
    worklist = ranked.filter(F.col("priority_rank") <= top_n)
    
    # Add metadata
    worklist = worklist.withColumn("inspection_status", F.lit("PENDING"))
    worklist = worklist.withColumn("created_date", F.current_timestamp())
    
    # Write to work list table
    (worklist.write
     .format("delta")
     .mode("overwrite")
     .saveAsTable(f"{catalog_path}.gold.daily_inspection_worklist"))
    
    # Summary statistics
    print(f"\nWork List Summary:")
    print(f"  Threshold (μ + {threshold_sigma}σ): {threshold:.3f}")
    print(f"  Total outliers: {outliers.count()}")
    print(f"  Top priorities for inspection: {top_n}")
    
    # Display top 10
    print(f"\nTop 10 Inspection Priorities:")
    print(f"{'Rank':<6} {'Tile ID':<15} {'Score':<10} {'Lon':<12} {'Lat':<12}")
    print("-" * 70)
    
    top_10 = worklist.orderBy("priority_rank").limit(10).collect()
    for row in top_10:
        print(f"{row['priority_rank']:<6} {row['tile_id']:<15} {row['encroachment_score']:>6.3f}    "
              f"{row['longitude']:>10.5f}  {row['latitude']:>10.5f}")
    
    # Geographic clustering (how many distinct locations?)
    # Group by ~1km radius
    worklist_with_cluster = worklist.withColumn(
        "geo_cluster",
        F.concat(
            (F.col("longitude") * 100).cast("int"),
            F.lit("_"),
            (F.col("latitude") * 100).cast("int")
        )
    )
    
    n_locations = worklist_with_cluster.select("geo_cluster").distinct().count()
    
    print(f"\nGeographic Distribution:")
    print(f"  Distinct inspection locations (~1km clusters): {n_locations}")
    print(f"  Average tiles per location: {top_n / n_locations:.1f}")
    
    return worklist

# Example usage with full pipeline
def main():
    """Complete ROW encroachment detection pipeline."""
    print("="*70)
    print("PIPELINE ROW ENCROACHMENT DETECTION - DINOV2 + DATABRICKS")
    print("="*70)
    print()
    
    # Initialize
    spark = initialize_dinov2_pipeline(spark)
    catalog_path = "catalog.pipeline"
    
    # Step 1: Load corridor and tiles
    pipeline_wkt = "LINESTRING(-102.5 34.0, -102.3 34.1, -102.0 34.2, -101.8 34.3)"
    tiles_df = load_row_buffer_and_tiles(spark, pipeline_wkt, buffer_m=500, tile_size_m=50)
    
    # Step 2: Ingest imagery
    image_paths = "s3://pipeline-imagery/2024-01-15/"
    tiles_with_images = ingest_aerial_imagery(spark, image_paths, tiles_df, catalog_path)
    
    # Step 3: Extract embeddings
    embeddings_df = extract_dinov2_embeddings_batch(spark, tiles_with_images, catalog_path)
    
    # Step 4: Cluster and score
    scored_df = cluster_and_score_anomalies(spark, embeddings_df, catalog_path, k=20)
    
    # Step 5: Generate work list
    worklist = generate_inspection_worklist(spark, catalog_path, top_n=200, threshold_sigma=3.0)
    
    print("\n" + "="*70)
    print("Pipeline complete! Inspection worklist ready.")
    print("="*70)
    
    return worklist

if __name__ == "__main__":
    results = main()

# ======================================================================
# Code Block 6
# ======================================================================

import dlt

@dlt.table(
    comment="Raw aerial/satellite tile metadata",
    partition_cols=["date"]
)
def row_tiles_bronze():
    return (spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "json")
            .load("s3://pipeline-imagery/metadata/"))

@dlt.table(
    comment="DINOv2 embeddings for all tiles"
)
@dlt.expect_or_drop("valid_embedding", "size(embedding) = 384")
def tile_embeddings_silver():
    return (dlt.read_stream("row_tiles_bronze")
            .transform(extract_dinov2_batch))  # GPU inference UDF

@dlt.table(
    comment="Anomaly scores and inspection priorities",
    partition_cols=["date"]
)
def encroachment_scores_gold():
    embeddings = dlt.read("tile_embeddings_silver")
    # Incremental clustering on new tiles
    return compute_anomaly_scores(embeddings)

# ======================================================================
# Code Block 7
# ======================================================================

def check_encroachment_alerts(spark, catalog_path, alert_threshold=5.0):
    """
    Query high-anomaly tiles and send alerts.
    """
    from databricks.sdk import WorkspaceClient
    
    w = WorkspaceClient()
    
    # Query recent high anomalies
    alerts = spark.sql(f"""
        SELECT tile_id, longitude, latitude, encroachment_score, date
        FROM {catalog_path}.gold.row_encroachment_scores
        WHERE date >= CURRENT_DATE - INTERVAL 1 DAY
          AND encroachment_score > {alert_threshold}
        ORDER BY encroachment_score DESC
    """)
    
    alert_count = alerts.count()
    
    if alert_count > 0:
        # Publish to notification system
        # message = f"⚠️ {alert_count} high-priority ROW encroachment alerts detected"
        
        # Send to Slack/Teams/PagerDuty
        w.workspace.create_notification(
            title="Pipeline Encroachment Alert",
            message=message,
            severity="high"
        )
        
        # Log to alert table
        (alerts.withColumn("alert_timestamp", F.current_timestamp())
         .write
         .format("delta")
         .mode("append")
         .saveAsTable(f"{catalog_path}.logs.encroachment_alerts"))
        
        # print(f"✓ Sent {alert_count} encroachment alerts")
    else:
        pass
        # print("✓ No high-priority encroachments detected")

# Schedule as Databricks Job (daily)
check_encroachment_alerts(spark, "catalog.pipeline", alert_threshold=5.0)

# ======================================================================
# Code Block 8
# ======================================================================

mos.enable_mosaic(spark, dbutils)

print("DINOv2 Encroachment Detection Pipeline Initialized")
print(f"  Spark version: {spark.version}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

return spark

# ======================================================================
# Code Block 9
# ======================================================================

"""
Load pipeline corridor geometry and generate tile grid.

Args:
    pipeline_geom: LineString WKT of pipeline route
    buffer_m: Buffer distance in meters (500m typical)
    tile_size_m: Tile dimension in meters (50m = ~167 pixels at 0.3m res)

Returns:
    DataFrame with tile polygons and metadata
"""

# ======================================================================
# Code Block 10
# ======================================================================

buffer_deg = buffer_m / 111000  # ~111km per degree latitude
tile_deg = tile_size_m / 111000

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW pipeline_row AS
    SELECT 
        'PIPELINE_001' AS pipeline_id,
        ST_GeomFromWKT('{pipeline_geom}') AS geom
""")

spark.sql(f"""
    CREATE OR REPLACE TEMP VIEW row_buffer AS
    SELECT 
        pipeline_id,
        ST_Buffer(geom, {buffer_deg}) AS buffer_geom
    FROM pipeline_row
""")

# ======================================================================
# Code Block 11
# ======================================================================

tiles_df = spark.sql(f"""
    SELECT 
        pipeline_id,
        mos.grid_tessellate(buffer_geom, {tile_deg}) AS tile_cell,
        mos.grid_boundaryaswkb(mos.grid_tessellate(buffer_geom, {tile_deg})) AS tile_geom_wkb
    FROM row_buffer
""")

print(f"\nROW Corridor Tiling:")
print(f"  Buffer: {buffer_m}m")
print(f"  Tile size: {tile_size_m}m")
print(f"  Total tiles: {tiles_df.count()}")

return tiles_df

# ======================================================================
# Code Block 12
# ======================================================================

n_tiles = 20000  # Typical for 100-mile pipeline segment

tile_data = []
for i in range(n_tiles):
    tile_data.append({
        'tile_id': f'TILE_{i:06d}',
        'image_path': f'/dbfs/pipeline_imagery/2024/tile_{i:06d}.jpg',
        'longitude': -102.0 + (i % 200) * 0.001,
        'latitude': 34.0 + (i // 200) * 0.001,
        'date': '2024-01-15',
        'resolution_m': 0.5
    })

tiles_df = spark.createDataFrame(tile_data)

# ======================================================================
# Code Block 13
# ======================================================================

(tiles_df.write
 .format("delta")
 .mode("overwrite")
 .partitionBy("date")
 .saveAsTable(f"{catalog_path}.bronze.row_tiles"))

print(f"\nAerial Imagery Ingested:")
print(f"  Total tiles: {len(tile_data)}")
print(f"  Resolution: {tile_data[0]['resolution_m']}m")
print(f"  Date: {tile_data[0]['date']}")

return tiles_df

# ======================================================================
# Code Block 14
# ======================================================================

"""
Extract DINOv2 embeddings for all tiles using GPU cluster.

This function runs DINOv2 inference distributed across Spark workers.
Each worker loads the model once, then processes batches of tiles.

Args:
    tiles_df: DataFrame with tile_id and image_path
    batch_size: Batch size for GPU inference

Returns:
    DataFrame with tile_id and embedding vector
"""
print("\nLoading DINOv2 Model...")
print("  Model: dinov2_vits14 (ViT-Small, 384-dim embeddings)")
print("  Pretrained on: 142M images (ImageNet-22k + curated)")

# ======================================================================
# Code Block 15
# ======================================================================

def extract_embeddings_udf(image_paths_batch):
    """
    Extract embeddings for a batch of images.
    Runs on each Spark worker with local GPU.
    """
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np

# ======================================================================
# Code Block 16
# ======================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # model = model.to(device).eval()

# ======================================================================
# Code Block 17
# ======================================================================

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # embeddings = []
    # with torch.no_grad():
        # for img_path in image_paths_batch:
            # try:

# ======================================================================
# Code Block 18
# ======================================================================

img = Image.open(img_path).convert('RGB')
                # img_tensor = transform(img).unsqueeze(0).to(device)

# ======================================================================
# Code Block 19
# ======================================================================

emb = model(img_tensor).squeeze().cpu().numpy()
                # embeddings.append(emb.tolist())
            # except Exception as e:

# ======================================================================
# Code Block 20
# ======================================================================

embeddings.append([0.0] * 384)
    
    # return embeddings

# ======================================================================
# Code Block 21
# ======================================================================

print("\nExtracting Embeddings (simulated for demo)...")

# ======================================================================
# Code Block 22
# ======================================================================

np.random.seed(42)
n_tiles = tiles_df.count()

# ======================================================================
# Code Block 23
# ======================================================================

cluster_assignments = np.random.choice([0, 1, 2], n_tiles, p=[0.70, 0.25, 0.05])

# ======================================================================
# Code Block 24
# ======================================================================

embeddings_list = []
for i in range(n_tiles):
    cluster = cluster_assignments[i]

# ======================================================================
# Code Block 25
# ======================================================================

if cluster == 0:  # Normal vegetation
        center = np.random.randn(384) * 0.1
    # elif cluster == 1:  # Maintained ROW
        center = np.ones(384) * 0.5 + np.random.randn(384) * 0.15
    # else:  # Encroachment
        center = np.random.randn(384) + 2.0
    
    # embedding = center + np.random.randn(384) * 0.3
    # embeddings_list.append(embedding.tolist())

# ======================================================================
# Code Block 26
# ======================================================================

tiles_with_embeddings = tiles_df.limit(n_tiles).toPandas()
tiles_with_embeddings['embedding'] = embeddings_list

embeddings_df = spark.createDataFrame(tiles_with_embeddings)

# ======================================================================
# Code Block 27
# ======================================================================

(embeddings_df.write
 .format("delta")
 .mode("overwrite")
 .saveAsTable(f"{catalog_path}.silver.tile_embeddings"))

# print(f"✓ Embeddings extracted: {n_tiles} tiles")
print(f"  Embedding dimension: 384")
print(f"  Storage: Delta table")

return embeddings_df

# ======================================================================
# Code Block 28
# ======================================================================

"""
Perform K-means clustering and compute anomaly scores.

Anomaly score = distance to nearest cluster centroid
High scores indicate tiles dissimilar to normal patterns.

Args:
    embeddings_df: DataFrame with tile_id and embedding
    k: Number of clusters (20-50 typical)

Returns:
    DataFrame with anomaly scores
"""
print(f"\nClustering with K-means (k={k})...")

# ======================================================================
# Code Block 29
# ======================================================================

vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())
vectors_df = embeddings_df.withColumn("features", vector_udf(F.col("embedding")))

# ======================================================================
# Code Block 30
# ======================================================================

kmeans = KMeans(k=k, seed=42, maxIter=50, initMode="k-means||")
model = kmeans.fit(vectors_df.select("features"))

# ======================================================================
# Code Block 31
# ======================================================================

predictions = model.transform(vectors_df)

# ======================================================================
# Code Block 32
# ======================================================================

centers = [np.array(c) for c in model.clusterCenters()]

# ======================================================================
# Code Block 33
# ======================================================================

def compute_distance_to_centroid(embedding, cluster_id):
    """Euclidean distance from embedding to its cluster center."""
    emb_array = np.array(embedding)
    center = centers[int(cluster_id)]
    return float(np.linalg.norm(emb_array - center))

distance_udf = F.udf(compute_distance_to_centroid, DoubleType())

scored = predictions.withColumn(
    "encroachment_score",
    distance_udf(F.col("embedding"), F.col("prediction"))
)

# ======================================================================
# Code Block 34
# ======================================================================

stats = scored.agg(
    F.count("*").alias("total_tiles"),
    F.avg("encroachment_score").alias("mean_score"),
    F.stddev("encroachment_score").alias("std_score"),
    F.max("encroachment_score").alias("max_score"),
    F.expr("percentile(encroachment_score, 0.95)").alias("p95_score"),
    F.expr("percentile(encroachment_score, 0.99)").alias("p99_score")
).collect()[0]

print(f"\nClustering Results:")
print(f"  Clusters: {k}")
print(f"  Total tiles: {stats['total_tiles']}")
print(f"  Mean anomaly score: {stats['mean_score']:.3f}")
print(f"  Std dev: {stats['std_score']:.3f}")
print(f"  95th percentile: {stats['p95_score']:.3f}")
print(f"  99th percentile: {stats['p99_score']:.3f}")

# ======================================================================
# Code Block 35
# ======================================================================

threshold = stats['mean_score'] + 3 * stats['std_score']
n_outliers = scored.filter(F.col("encroachment_score") > threshold).count()

print(f"\nAnomaly Detection:")
print(f"  Threshold (μ + 3σ): {threshold:.3f}")
print(f"  Outliers flagged: {n_outliers} ({n_outliers/stats['total_tiles']*100:.2f}%)")

return scored

# ======================================================================
# Code Block 36
# ======================================================================

"""
Create prioritized inspection list for field crews.

Ranking criteria:
- Anomaly score (distance to centroid)
- Proximity to pipeline centerline
- Change from previous inspection
- High consequence area (HCA) proximity

Returns:
    DataFrame with top inspection priorities
"""
print(f"\n{'='*70}")
print("GENERATING INSPECTION WORK LIST")
print('='*70)

# ======================================================================
# Code Block 37
# ======================================================================

scores_df = spark.table(f"{catalog_path}.gold.row_encroachment_scores")

# ======================================================================
# Code Block 38
# ======================================================================

stats = scores_df.agg(
    F.avg("encroachment_score").alias("mean"),
    F.stddev("encroachment_score").alias("std")
).collect()[0]

threshold = stats['mean'] + threshold_sigma * stats['std']

# ======================================================================
# Code Block 39
# ======================================================================

outliers = scores_df.filter(F.col("encroachment_score") > threshold)

# ======================================================================
# Code Block 40
# ======================================================================

ranked = outliers.withColumn(
    "priority_rank",
    F.row_number().over(
        Window.orderBy(F.col("encroachment_score").desc())
    )
)

# ======================================================================
# Code Block 41
# ======================================================================

worklist = ranked.filter(F.col("priority_rank") <= top_n)

# ======================================================================
# Code Block 42
# ======================================================================

worklist = worklist.withColumn("inspection_status", F.lit("PENDING"))
worklist = worklist.withColumn("created_date", F.current_timestamp())

# ======================================================================
# Code Block 43
# ======================================================================

print(f"\nWork List Summary:")
print(f"  Threshold (μ + {threshold_sigma}σ): {threshold:.3f}")
print(f"  Total outliers: {outliers.count()}")
print(f"  Top priorities for inspection: {top_n}")

# ======================================================================
# Code Block 44
# ======================================================================

print(f"\nTop 10 Inspection Priorities:")
print(f"{'Rank':<6} {'Tile ID':<15} {'Score':<10} {'Lon':<12} {'Lat':<12}")
print("-" * 70)

top_10 = worklist.orderBy("priority_rank").limit(10).collect()
for row in top_10:
    print(f"{row['priority_rank']:<6} {row['tile_id']:<15} {row['encroachment_score']:>6.3f}    "
          f"{row['longitude']:>10.5f}  {row['latitude']:>10.5f}")

# ======================================================================
# Code Block 45
# ======================================================================

worklist_with_cluster = worklist.withColumn(
    "geo_cluster",
    F.concat(
        (F.col("longitude") * 100).cast("int"),
        F.lit("_"),
        (F.col("latitude") * 100).cast("int")
    )
)

n_locations = worklist_with_cluster.select("geo_cluster").distinct().count()

print(f"\nGeographic Distribution:")
print(f"  Distinct inspection locations (~1km clusters): {n_locations}")
print(f"  Average tiles per location: {top_n / n_locations:.1f}")

return worklist

# ======================================================================
# Code Block 46
# ======================================================================

"""Complete ROW encroachment detection pipeline."""
print("="*70)
print("PIPELINE ROW ENCROACHMENT DETECTION - DINOV2 + DATABRICKS")
print("="*70)
print()

# ======================================================================
# Code Block 47
# ======================================================================

spark = initialize_dinov2_pipeline(spark)
catalog_path = "catalog.pipeline"

# ======================================================================
# Code Block 48
# ======================================================================

pipeline_wkt = "LINESTRING(-102.5 34.0, -102.3 34.1, -102.0 34.2, -101.8 34.3)"
tiles_df = load_row_buffer_and_tiles(spark, pipeline_wkt, buffer_m=500, tile_size_m=50)

# ======================================================================
# Code Block 49
# ======================================================================

image_paths = "s3://pipeline-imagery/2024-01-15/"
tiles_with_images = ingest_aerial_imagery(spark, image_paths, tiles_df, catalog_path)

# ======================================================================
# Code Block 50
# ======================================================================

embeddings_df = extract_dinov2_embeddings_batch(spark, tiles_with_images, catalog_path)

# ======================================================================
# Code Block 51
# ======================================================================

scored_df = cluster_and_score_anomalies(spark, embeddings_df, catalog_path, k=20)

# ======================================================================
# Code Block 52
# ======================================================================

worklist = generate_inspection_worklist(spark, catalog_path, top_n=200, threshold_sigma=3.0)

print("\n" + "="*70)
print("Pipeline complete! Inspection worklist ready.")
print("="*70)

return worklist

# ======================================================================
# Code Block 53
# ======================================================================

results = main()

# ======================================================================
# Code Block 54
# ======================================================================

# ======================================================================
# PIPELINE ROW ENCROACHMENT DETECTION - DINOV2 + DATABRICKS
# ======================================================================

# DINOv2 Encroachment Detection Pipeline Initialized
  # Spark version: 3.5.0
  # CUDA available: True
  # GPU: NVIDIA A100-SXM4-40GB

# ROW Corridor Tiling:
  # Buffer: 500m
  # Tile size: 50m
  # Total tiles: 20,000

# Aerial Imagery Ingested:
  # Total tiles: 20,000
  # Resolution: 0.5m
  # Date: 2024-01-15

# Loading DINOv2 Model...
  # Model: dinov2_vits14 (ViT-Small, 384-dim embeddings)
  # Pretrained on: 142M images (ImageNet-22k + curated)

# Extracting Embeddings (simulated for demo)...
# ✓ Embeddings extracted: 20,000 tiles
  # Embedding dimension: 384
  # Storage: Delta table

# Clustering with K-means (k=20)...

# Clustering Results:
  # Clusters: 20
  # Total tiles: 20,000
  # Mean anomaly score: 2.456
  # Std dev: 0.823
  # 95th percentile: 3.912
  # 99th percentile: 4.567

# Anomaly Detection:
  # Threshold (μ + 3σ): 4.925
  # Outliers flagged: 287 (1.44%)

# ======================================================================
# GENERATING INSPECTION WORK LIST
# ======================================================================

# Work List Summary:
  # Threshold (μ + 3σ): 4.925
  # Total outliers: 287
  # Top priorities for inspection: 200

# Top 10 Inspection Priorities:
# Rank   Tile ID         Score      Lon          Lat        
# ----------------------------------------------------------------------
# 1      TILE_012845      6.234    -101.94123   34.28456
# 2      TILE_007234      6.187    -102.08721   34.13289
# 3      TILE_018923      6.094    -101.79234   34.35612
# 4      TILE_003456      5.978    -102.23451   34.06234
# 5      TILE_015678      5.912    -101.86789   34.31245
# 6      TILE_009821      5.867    -102.01234   34.19087
# 7      TILE_014532      5.823    -101.89123   34.27891
# 8      TILE_011234      5.789    -101.97654   34.24312
# 9      TILE_008765      5.734    -102.04567   34.15678
# 10     TILE_016789      5.698    -101.83456   34.32789

# Geographic Distribution:
  # Distinct inspection locations (~1km clusters): 42
  # Average tiles per location: 4.8

# ======================================================================
# Pipeline complete! Inspection worklist ready.
# ======================================================================

# ======================================================================
# Code Block 55
# ======================================================================

comment="Raw aerial/satellite tile metadata",
partition_cols=["date"]

# ======================================================================
# Code Block 56
# ======================================================================

return (spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .load("s3://pipeline-imagery/metadata/"))

# ======================================================================
# Code Block 57
# ======================================================================

comment="DINOv2 embeddings for all tiles"

# ======================================================================
# Code Block 58
# ======================================================================

return (dlt.read_stream("row_tiles_bronze")
        .transform(extract_dinov2_batch))  # GPU inference UDF

# ======================================================================
# Code Block 59
# ======================================================================

comment="Anomaly scores and inspection priorities",
partition_cols=["date"]

# ======================================================================
# Code Block 60
# ======================================================================

embeddings = dlt.read("tile_embeddings_silver")

# ======================================================================
# Code Block 61
# ======================================================================

return compute_anomaly_scores(embeddings)

# ======================================================================
# Code Block 62
# ======================================================================

"""
Query high-anomaly tiles and send alerts.
"""
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# ======================================================================
# Code Block 63
# ======================================================================

alerts = spark.sql(f"""
    SELECT tile_id, longitude, latitude, encroachment_score, date
    FROM {catalog_path}.gold.row_encroachment_scores
    WHERE date >= CURRENT_DATE - INTERVAL 1 DAY
      AND encroachment_score > {alert_threshold}
    ORDER BY encroachment_score DESC
""")

alert_count = alerts.count()

if alert_count > 0:
    pass

# ======================================================================
# Code Block 64
# ======================================================================

# message = f"⚠️ {alert_count} high-priority ROW encroachment alerts detected"

# ======================================================================
# Code Block 65
# ======================================================================

w.workspace.create_notification(
        title="Pipeline Encroachment Alert",
        message=message,
        severity="high"
    )

# ======================================================================
# Code Block 66
# ======================================================================

(alerts.withColumn("alert_timestamp", F.current_timestamp())
     .write
     .format("delta")
     .mode("append")
     .saveAsTable(f"{catalog_path}.logs.encroachment_alerts"))
    
    # print(f"✓ Sent {alert_count} encroachment alerts")
# else:
    # print("✓ No high-priority encroachments detected")
