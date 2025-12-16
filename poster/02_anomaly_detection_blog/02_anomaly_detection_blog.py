#!/usr/bin/env python3
"""
Python code extracted from 02_anomaly_detection_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import numpy as np

# Load plant data
plants = pd.read_parquet('egrid_all_plants_1996-2023.parquet')
plants_2023 = plants[plants['data_year'] == 2023].copy()

# Create features for anomaly detection
plants_2023['carbon_intensity'] = (
    plants_2023['Plant annual CO2 emissions (tons)'] / 
    plants_2023['Plant annual net generation (MWh)']
)

plants_2023['capacity_factor'] = (
    plants_2023['Plant annual net generation (MWh)'] / 
    (plants_2023['Plant nameplate capacity (MW)'] * 8760)
)

# Log transforms for skewed distributions
plants_2023['log_generation'] = np.log1p(plants_2023['Plant annual net generation (MWh)'])
plants_2023['log_co2'] = np.log1p(plants_2023['Plant annual CO2 emissions (tons)'])

print(f"Analyzing {len(plants_2023):,} plants")
print(f"Features: carbon_intensity, capacity_factor, log_generation, log_co2")

# ======================================================================
# Code Block 2
# ======================================================================

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Prepare and scale features
features = ['log_generation', 'log_co2', 'carbon_intensity', 'capacity_factor']
X = plants_2023[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expect 5% anomalies
    random_state=42
)

predictions = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

plants_2023.loc[X.index, 'iso_anomaly'] = (predictions == -1)
plants_2023.loc[X.index, 'iso_score'] = anomaly_scores

print(f"Detected {(predictions == -1).sum()} anomalies ({(predictions == -1).sum()/len(predictions)*100:.1f}%)")

# ======================================================================
# Code Block 3
# ======================================================================

from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

lof_predictions = lof.fit_predict(X_scaled)
lof_scores = lof.negative_outlier_factor_

plants_2023.loc[X.index, 'lof_anomaly'] = (lof_predictions == -1)
plants_2023.loc[X.index, 'lof_score'] = lof_scores

print(f"LOF detected {(lof_predictions == -1).sum()} anomalies")

# ======================================================================
# Code Block 4
# ======================================================================

from tensorflow import keras
from tensorflow.keras import layers

# Build autoencoder
input_dim = X_scaled.shape[1]
encoding_dim = 2  # Compress to 2 dimensions

autoencoder = keras.Sequential([
    # Encoder
    layers.Dense(8, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.2),
    layers.Dense(4, activation='relu'),
    layers.Dense(encoding_dim, activation='relu'),
    
    # Decoder
    layers.Dense(4, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(8, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train (autoencoders learn normal patterns)
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=32, 
                validation_split=0.2, verbose=0)

# Calculate reconstruction errors
reconstructions = autoencoder.predict(X_scaled)
reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

# Threshold at 95th percentile
threshold = np.percentile(reconstruction_errors, 95)
ae_predictions = (reconstruction_errors > threshold)

plants_2023.loc[X.index, 'ae_anomaly'] = ae_predictions
plants_2023.loc[X.index, 'ae_error'] = reconstruction_errors

print(f"Autoencoder detected {ae_predictions.sum()} anomalies")
print(f"Threshold: {threshold:.4f}")

# ======================================================================
# Code Block 5
# ======================================================================

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
cluster_labels = dbscan.fit_predict(X_scaled)

# Label -1 means noise (anomaly)
dbscan_anomalies = (cluster_labels == -1)
plants_2023.loc[X.index, 'dbscan_anomaly'] = dbscan_anomalies

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"DBSCAN found {n_clusters} clusters")
print(f"Noise points (anomalies): {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

# ======================================================================
# Code Block 6
# ======================================================================

from scipy import stats

def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    return z_scores > threshold

# Apply to each feature
for feature in ['carbon_intensity', 'capacity_factor', 'log_generation']:
    outliers = detect_outliers_zscore(plants_2023[feature])
    plants_2023[f'{feature}_zscore_outlier'] = outliers
    print(f"{feature}: {outliers.sum()} outliers ({outliers.sum()/len(outliers)*100:.1f}%)")

# ======================================================================
# Code Block 7
# ======================================================================

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)

# ======================================================================
# Code Block 8
# ======================================================================

def detect_outliers_mad(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return np.abs(modified_z) > threshold

# ======================================================================
# Code Block 9
# ======================================================================

# Create voting system
anomaly_votes = (
    plants_2023['iso_anomaly'].astype(int) +
    plants_2023['lof_anomaly'].astype(int) +
    plants_2023['ae_anomaly'].astype(int) +
    plants_2023['dbscan_anomaly'].astype(int) +
    (plants_2023['carbon_intensity_zscore_outlier'].astype(int))
)

plants_2023['ensemble_votes'] = anomaly_votes
plants_2023['ensemble_anomaly'] = anomaly_votes >= 2  # 2+ methods agree

print("Ensemble Results:")
print(f"  2+ votes: {(anomaly_votes >= 2).sum()} plants")
print(f"  3+ votes: {(anomaly_votes >= 3).sum()} plants (high confidence)")
print(f"  4+ votes: {(anomaly_votes >= 4).sum()} plants (very high confidence)")
print(f"  All 5 agree: {(anomaly_votes == 5).sum()} plants (extreme outliers)")

# ======================================================================
# Code Block 10
# ======================================================================

def analyze_anomaly(plant_row, population_stats):
    """Identify which features are most unusual"""
    contributions = {}
    for feature in features:
        mean = population_stats[feature]['mean']
        std = population_stats[feature]['std']
        z_score = abs((plant_row[feature] - mean) / std)
        contributions[feature] = z_score
    return contributions

# Analyze top anomalies
high_confidence = plants_2023[plants_2023['ensemble_votes'] >= 3]

for idx, row in high_confidence.head(10).iterrows():
    print(f"\nPlant: {row.get('Plant name', 'Unknown')}")
    print(f"  Ensemble votes: {row['ensemble_votes']}")
    
    contribs = analyze_anomaly(row, population_stats)
    sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    
    print("  Most unusual features:")
    for feature, z_score in sorted_contribs[:3]:
        print(f"    {feature}: {z_score:.2f} std deviations")

# ======================================================================
# Code Block 11
# ======================================================================

n_estimators=100,
contamination=0.05,  # Expect 5% anomalies
random_state=42

# ======================================================================
# Code Block 12
# ======================================================================

n_neighbors=20,
contamination=0.05

# ======================================================================
# Code Block 13
# ======================================================================

layers.Dense(8, activation='relu', input_shape=(input_dim,)),
layers.Dropout(0.2),
layers.Dense(4, activation='relu'),
layers.Dense(encoding_dim, activation='relu'),

# ======================================================================
# Code Block 14
# ======================================================================

layers.Dense(4, activation='relu'),
layers.Dropout(0.2),
layers.Dense(8, activation='relu'),
layers.Dense(input_dim, activation='linear')

# ======================================================================
# Code Block 15
# ======================================================================

validation_split=0.2, verbose=0)

# ======================================================================
# Code Block 16
# ======================================================================

z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
return z_scores > threshold

# ======================================================================
# Code Block 17
# ======================================================================

outliers = detect_outliers_zscore(plants_2023[feature])
plants_2023[f'{feature}_zscore_outlier'] = outliers
print(f"{feature}: {outliers.sum()} outliers ({outliers.sum()/len(outliers)*100:.1f}%)")

# ======================================================================
# Code Block 18
# ======================================================================

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
return (data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)

# ======================================================================
# Code Block 19
# ======================================================================

median = np.median(data)
mad = np.median(np.abs(data - median))
modified_z = 0.6745 * (data - median) / mad
return np.abs(modified_z) > threshold

# ======================================================================
# Code Block 20
# ======================================================================

"""Identify which features are most unusual"""
contributions = {}
for feature in features:
    mean = population_stats[feature]['mean']
    std = population_stats[feature]['std']
    z_score = abs((plant_row[feature] - mean) / std)
    contributions[feature] = z_score
return contributions

# ======================================================================
# Code Block 21
# ======================================================================

print(f"\nPlant: {row.get('Plant name', 'Unknown')}")
print(f"  Ensemble votes: {row['ensemble_votes']}")

contribs = analyze_anomaly(row, population_stats)
sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)

print("  Most unusual features:")
for feature, z_score in sorted_contribs[:3]:
    print(f"    {feature}: {z_score:.2f} std deviations")
