#!/usr/bin/env python3
"""
Python code extracted from 22_ansys_fea_databricks_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

from ansys.mapdl.reader import read_binary
import numpy as np

def von_mises(sx, sy, sz, txy, tyz, txz):
    """Compute von Mises stress from stress tensor components."""
    a = (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
    b = 6.0 * (txy**2 + tyz**2 + txz**2)
    return np.sqrt(0.5 * (a + b))

# Read result file
res = read_binary('/dbfs/FileStore/fea/model.rst')
nnum = res.nnum  # Node numbers
n_sets = res.nsets  # Number of result sets (load steps)

# Extract nodal displacement for set 0
disp = res.nodal_displacement(0)  # Shape: [n_nodes, 3] (ux, uy, uz)

# Extract nodal stress for set 0
stress = res.nodal_stress(0)  # Shape: [n_nodes, 6] (sx, sy, sz, txy, tyz, txz)

# Compute von Mises stress
vm = von_mises(stress[:, 0], stress[:, 1], stress[:, 2], 
               stress[:, 3], stress[:, 4], stress[:, 5])

# ======================================================================
# Code Block 2
# ======================================================================

import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Create catalog and schema
catalog = 'fea'
schema = 'ansys'
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# Build DataFrame
stress_records = []
for i in range(n_sets):
    stress = res.nodal_stress(i)
    vm = von_mises(stress[:, 0], stress[:, 1], stress[:, 2], 
                   stress[:, 3], stress[:, 4], stress[:, 5])
    for j, node in enumerate(nnum):
        stress_records.append({
            'source_file': 'model.rst',
            'set_idx': i,
            'node': int(node),
            'sx': float(stress[j, 0]),
            'sy': float(stress[j, 1]),
            'sz': float(stress[j, 2]),
            'txy': float(stress[j, 3]),
            'tyz': float(stress[j, 4]),
            'txz': float(stress[j, 5]),
            'von_mises': float(vm[j])
        })

pdf = pd.DataFrame(stress_records)
sdf = spark.createDataFrame(pdf)
sdf.write.mode('overwrite').saveAsTable(f'{catalog}.{schema}.nodal_stress')

# ======================================================================
# Code Block 3
# ======================================================================

import matplotlib.pyplot as plt
import pandas as pd

# Query stress data for set_idx = 0
query = """
SELECT node, von_mises 
FROM fea.ansys.nodal_stress 
WHERE set_idx = 0 
ORDER BY node
"""
pdf = spark.sql(query).toPandas()

# Configure minimalist style
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(pdf['node'], pdf['von_mises'] / 1e6, 
        color='black', linewidth=1, marker='o', markersize=2)

ax.set_xlabel('Node ID', fontsize=11)
ax.set_ylabel('von Mises Stress (MPa)', fontsize=11)
ax.set_title('Nodal Stress Distribution (Load Case 0)', fontsize=12, pad=15)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('von_mises_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 4
# ======================================================================

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler

# Aggregate per simulation
agg_query = """
SELECT 
    REGEXP_EXTRACT(source_file, 'E(\d+)', 1) AS elastic_modulus,
    REGEXP_EXTRACT(source_file, 'F(\d+)', 1) AS force,
    MAX(von_mises) AS peak_stress
FROM fea.ansys.nodal_stress
GROUP BY source_file
"""
df = spark.sql(agg_query)

# Feature engineering
assembler = VectorAssembler(
    inputCols=['elastic_modulus', 'force'], 
    outputCol='features'
)
df = assembler.transform(df)

# Split and train
train, test = df.randomSplit([0.8, 0.2], seed=42)
rf = RandomForestRegressor(
    featuresCol='features', 
    labelCol='peak_stress', 
    numTrees=100
)
model = rf.fit(train)

# Evaluate
predictions = model.transform(test)
predictions.select('elastic_modulus', 'force', 'peak_stress', 'prediction').show(10)

# ======================================================================
# Code Block 5
# ======================================================================

nodes = res.mesh.nodes  # Shape: [n_nodes, 3] (x, y, z)
node_coords = pd.DataFrame({
    'node': nnum,
    'x': nodes[:, 0],
    'y': nodes[:, 1],
    'z': nodes[:, 2]
})
spark.createDataFrame(node_coords).write.mode('overwrite').saveAsTable('fea.ansys.node_coords')

# ======================================================================
# Code Block 6
# ======================================================================

import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Query nodal stress and coordinates
query = """
SELECT n.x, n.y, st.von_mises
FROM fea.ansys.node_coords n
JOIN fea.ansys.nodal_stress st ON n.node = st.node
WHERE st.set_idx = 0
"""
pdf = spark.sql(query).toPandas()

# Create grid
xi = np.linspace(pdf['x'].min(), pdf['x'].max(), 200)
yi = np.linspace(pdf['y'].min(), pdf['y'].max(), 200)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate stress
Zi = griddata((pdf['x'], pdf['y']), pdf['von_mises'], (Xi, Yi), method='cubic')

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(Xi, Yi, Zi / 1e6, levels=20, cmap='viridis')
cbar = plt.colorbar(contour, ax=ax, label='von Mises (MPa)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Stress Contour (Load Case 0)')
plt.savefig('stress_contour.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 7
# ======================================================================

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
catalog = 'fea'
schema = 'ansys'
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

# ======================================================================
# Code Block 8
# ======================================================================

# Databricks notebook: Ansys FEA Ingestion and Analysis
# Prereqs:
# 1. Upload .rst files to dbfs:/FileStore/fea/
# 2. Attach to cluster with DBR 14+

# COMMAND ----------
# Install dependencies
# %pip install -q ansys-mapdl-reader pyvista matplotlib pandas
dbutils.library.restartPython()

# COMMAND ----------
# Configuration
from pyspark.sql import SparkSession
import os
import pandas as pd
import numpy as np
from ansys.mapdl.reader import read_binary

spark = SparkSession.builder.getOrCreate()

INPUT_DIR = 'dbfs:/FileStore/fea'
CATALOG = 'fea'
SCHEMA = 'ansys'
TABLE_STRESS = f'{CATALOG}.{SCHEMA}.nodal_stress'
TABLE_DISP = f'{CATALOG}.{SCHEMA}.nodal_displacement'

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------
# Helper functions
def von_mises(sx, sy, sz, txy, tyz, txz):
    """Compute von Mises stress from tensor components."""
    a = (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
    b = 6.0 * (txy**2 + tyz**2 + txz**2)
    return np.sqrt(0.5 * (a + b))

def list_rst_files(dbfs_dir):
    """List all .rst files in DBFS directory."""
    return [f.path for f in dbutils.fs.ls(dbfs_dir) 
            if f.path.lower().endswith('.rst')]

# COMMAND ----------
# Ingest all .rst files
rst_paths = list_rst_files(INPUT_DIR)
print(f'Found {len(rst_paths)} .rst files')

all_stress = []
all_disp = []

for dbfs_path in rst_paths:
    # Convert dbfs:/ to local /dbfs/
    local_path = '/dbfs' + dbfs_path[len('dbfs:'):]
    filename = os.path.basename(local_path)
    
    res = read_binary(local_path)
    nnum = res.nnum
    n_sets = res.nsets
    
    for i in range(n_sets):
        # Displacement
        disp = res.nodal_displacement(i)
        disp_records = pd.DataFrame({
            'source_file': filename,
            'set_idx': i,
            'node': nnum,
            'ux': disp[:, 0],
            'uy': disp[:, 1],
            'uz': disp[:, 2]
        })
        all_disp.append(disp_records)
        
        # Stress
        stress = res.nodal_stress(i)
        vm = von_mises(stress[:, 0], stress[:, 1], stress[:, 2],
                       stress[:, 3], stress[:, 4], stress[:, 5])
        stress_records = pd.DataFrame({
            'source_file': filename,
            'set_idx': i,
            'node': nnum,
            'sx': stress[:, 0],
            'sy': stress[:, 1],
            'sz': stress[:, 2],
            'txy': stress[:, 3],
            'tyz': stress[:, 4],
            'txz': stress[:, 5],
            'von_mises': vm
        })
        all_stress.append(stress_records)

# Write to Delta
if all_disp:
    disp_df = pd.concat(all_disp, ignore_index=True)
    spark.createDataFrame(disp_df).write.mode('overwrite').saveAsTable(TABLE_DISP)
    # print(f' Wrote {len(disp_df):,} displacement records to {TABLE_DISP}')

if all_stress:
    stress_df = pd.concat(all_stress, ignore_index=True)
    spark.createDataFrame(stress_df).write.mode('overwrite').saveAsTable(TABLE_STRESS)
    # print(f' Wrote {len(stress_df):,} stress records to {TABLE_STRESS}')

# COMMAND ----------
# Query: Find high-stress nodes
high_stress = spark.sql(f"""
SELECT 
    node,
    AVG(von_mises) AS avg_vm,
    MAX(von_mises) AS max_vm,
    COUNT(*) AS n_cases
FROM {TABLE_STRESS}
GROUP BY node
ORDER BY max_vm DESC
LIMIT 20
""")
high_stress.show()

# COMMAND ----------
# Visualization: Von Mises profile
import matplotlib.pyplot as plt

pdf = spark.sql(f"""
SELECT node, von_mises
FROM {TABLE_STRESS}
WHERE set_idx = 0
ORDER BY node
""").toPandas()

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(pdf['node'], pdf['von_mises'] / 1e6, 
        color='black', linewidth=1, marker='o', markersize=2)
ax.set_xlabel('Node ID', fontsize=11)
ax.set_ylabel('von Mises Stress (MPa)', fontsize=11)
ax.set_title('Nodal Stress Distribution (Load Case 0)', fontsize=12, pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
plt.tight_layout()
plt.savefig('/dbfs/FileStore/von_mises_profile.png', dpi=300, bbox_inches='tight')
plt.show()
# print(' Saved visualization')

# COMMAND ----------
# ML: Train surrogate model
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler

# Aggregate per simulation
agg_df = spark.sql(f"""
SELECT 
    source_file,
    AVG(von_mises) AS avg_stress,
    MAX(von_mises) AS max_stress,
    STDDEV(von_mises) AS std_stress
FROM {TABLE_STRESS}
GROUP BY source_file
""")

# Placeholder: extract design parameters from filename
# In production, join with metadata table
assembler = VectorAssembler(
    inputCols=['avg_stress', 'std_stress'],
    outputCol='features'
)
ml_df = assembler.transform(agg_df)

train, test = ml_df.randomSplit([0.8, 0.2], seed=42)
rf = RandomForestRegressor(
    featuresCol='features',
    labelCol='max_stress',
    numTrees=50
)
model = rf.fit(train)

predictions = model.transform(test)
predictions.select('source_file', 'max_stress', 'prediction').show(10)
# print(' Surrogate model trained')

# ======================================================================
# Code Block 9
# ======================================================================

"""Compute von Mises stress from stress tensor components."""
a = (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
b = 6.0 * (txy**2 + tyz**2 + txz**2)
return np.sqrt(0.5 * (a + b))

# ======================================================================
# Code Block 10
# ======================================================================

stress = res.nodal_stress(i)
vm = von_mises(stress[:, 0], stress[:, 1], stress[:, 2], 
               stress[:, 3], stress[:, 4], stress[:, 5])
for j, node in enumerate(nnum):
    stress_records.append({
        'source_file': 'model.rst',
        'set_idx': i,
        'node': int(node),
        'sx': float(stress[j, 0]),
        'sy': float(stress[j, 1]),
        'sz': float(stress[j, 2]),
        'txy': float(stress[j, 3]),
        'tyz': float(stress[j, 4]),
        'txz': float(stress[j, 5]),
        'von_mises': float(vm[j])
    })

# ======================================================================
# Code Block 11
# ======================================================================

# color='black', linewidth=1, marker='o', markersize=2

# ======================================================================
# Code Block 12
# ======================================================================

inputCols=['elastic_modulus', 'force'], 
outputCol='features'

# ======================================================================
# Code Block 13
# ======================================================================

featuresCol='features', 
labelCol='peak_stress', 
numTrees=100

# ======================================================================
# Code Block 14
# ======================================================================

"""Compute von Mises stress from tensor components."""
a = (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
b = 6.0 * (txy**2 + tyz**2 + txz**2)
return np.sqrt(0.5 * (a + b))

# ======================================================================
# Code Block 15
# ======================================================================

"""List all .rst files in DBFS directory."""
return [f.path for f in dbutils.fs.ls(dbfs_dir) 
        if f.path.lower().endswith('.rst')]

# ======================================================================
# Code Block 16
# ======================================================================

local_path = '/dbfs' + dbfs_path[len('dbfs:'):]
filename = os.path.basename(local_path)

res = read_binary(local_path)
nnum = res.nnum
n_sets = res.nsets

for i in range(n_sets):

# ======================================================================
# Code Block 17
# ======================================================================

    pass
disp = res.nodal_displacement(i)
disp_records = pd.DataFrame({
        'source_file': filename,
        'set_idx': i,
        'node': nnum,
        'ux': disp[:, 0],
        'uy': disp[:, 1],
        'uz': disp[:, 2]
    })
all_disp.append(disp_records)

# ======================================================================
# Code Block 18
# ======================================================================

stress = res.nodal_stress(i)
vm = von_mises(stress[:, 0], stress[:, 1], stress[:, 2],
                   stress[:, 3], stress[:, 4], stress[:, 5])
stress_records = pd.DataFrame({
        'source_file': filename,
        'set_idx': i,
        'node': nnum,
        'sx': stress[:, 0],
        'sy': stress[:, 1],
        'sz': stress[:, 2],
        'txy': stress[:, 3],
        'tyz': stress[:, 4],
        'txz': stress[:, 5],
        'von_mises': vm
    })
all_stress.append(stress_records)

# ======================================================================
# Code Block 19
# ======================================================================

disp_df = pd.concat(all_disp, ignore_index=True)
spark.createDataFrame(disp_df).write.mode('overwrite').saveAsTable(TABLE_DISP)
# print(f' Wrote {len(disp_df):,} displacement records to {TABLE_DISP}')

# ======================================================================
# Code Block 20
# ======================================================================

stress_df = pd.concat(all_stress, ignore_index=True)
spark.createDataFrame(stress_df).write.mode('overwrite').saveAsTable(TABLE_STRESS)
# print(f' Wrote {len(stress_df):,} stress records to {TABLE_STRESS}')

# ======================================================================
# Code Block 21
# ======================================================================

# color='black', linewidth=1, marker='o', markersize=2

# ======================================================================
# Code Block 22
# ======================================================================

inputCols=['avg_stress', 'std_stress'],
outputCol='features'

# ======================================================================
# Code Block 23
# ======================================================================

featuresCol='features',
labelCol='max_stress',
numTrees=50
