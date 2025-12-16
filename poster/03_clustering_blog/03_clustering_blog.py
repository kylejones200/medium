#!/usr/bin/env python3
"""
Python code extracted from 03_clustering_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import numpy as np

plants = pd.read_parquet('egrid_all_plants_1996-2023.parquet')
plants_2023 = plants[plants['data_year'] == 2023].copy()

# Create clustering features
plants_2023['log_generation'] = np.log1p(
    plants_2023['Plant annual net generation (MWh)']
)

plants_2023['log_co2'] = np.log1p(
    plants_2023['Plant annual CO2 emissions (tons)']
)

plants_2023['carbon_intensity'] = (
    plants_2023['Plant annual CO2 emissions (tons)'] / 
    plants_2023['Plant annual net generation (MWh)']
)

plants_2023['capacity_factor'] = (
    plants_2023['Plant annual net generation (MWh)'] / 
    (plants_2023['Plant nameplate capacity (MW)'] * 8760)
)

features = ['log_generation', 'log_co2', 'carbon_intensity', 
            'capacity_factor', 'nox_intensity', 'so2_intensity']

X = plants_2023[features].dropna()
print(f"Clustering {len(X):,} plants on {len(features)} features")

# ======================================================================
# Code Block 2
# ======================================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test K from 2 to 10
results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    results.append({
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X_scaled, labels),
        'calinski': calinski_harabasz_score(X_scaled, labels)
    })
    print(f"K={k}: Silhouette={results[-1]['silhouette']:.3f}")

# ======================================================================
# Code Block 3
# ======================================================================

optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

plants_2023.loc[X.index, 'cluster'] = cluster_labels

print("\nCluster sizes:")
print(pd.Series(cluster_labels).value_counts().sort_index())

# ======================================================================
# Code Block 4
# ======================================================================

# Calculate cluster statistics
cluster_profiles = plants_2023.groupby('cluster').agg({
    'log_generation': 'median',
    'carbon_intensity': 'median',
    'capacity_factor': 'median',
    'Plant nameplate capacity (MW)': 'median'
}).round(3)

print("\nCluster Profiles:")
print(cluster_profiles)

# ======================================================================
# Code Block 5
# ======================================================================

from sklearn.mixture import GaussianMixture

# Find optimal number of components using BIC
bic_scores = []
for n in range(2, 11):
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    print(f"n={n}: BIC={gmm.bic(X_scaled):.1f}")

optimal_n = range(2, 11)[np.argmin(bic_scores)]
print(f"\nOptimal components: {optimal_n}")

# Train final GMM
gmm_final = GaussianMixture(
    n_components=5, 
    covariance_type='full',
    random_state=42
)
gmm_labels = gmm_final.fit_predict(X_scaled)
gmm_probas = gmm_final.predict_proba(X_scaled)

plants_2023.loc[X.index, 'gmm_cluster'] = gmm_labels
plants_2023.loc[X.index, 'gmm_probability'] = gmm_probas.max(axis=1)

# ======================================================================
# Code Block 6
# ======================================================================

# Find plants with uncertain membership
uncertain = plants_2023[plants_2023['gmm_probability'] < 0.7]
print(f"\nPlants with uncertain cluster membership: {len(uncertain)}")
print("These plants have characteristics of multiple clusters")

# Example: A plant that's 40% Cluster 1, 35% Cluster 2, 25% Cluster 3
# This might be a gas plant with significant renewable co-generation

# ======================================================================
# Code Block 7
# ======================================================================

import hdbscan

hdb = hdbscan.HDBSCAN(
    min_cluster_size=50,  # Minimum 50 plants per cluster
    min_samples=10,
    metric='euclidean'
)

hdb_labels = hdb.fit_predict(X_scaled)

n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
n_noise = list(hdb_labels).count(-1)

print(f"HDBSCAN found {n_clusters} natural clusters")
print(f"Outliers/noise: {n_noise} plants ({n_noise/len(hdb_labels)*100:.1f}%)")

# ======================================================================
# Code Block 8
# ======================================================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plants_2023.loc[X.index, 'pca1'] = X_pca[:, 0]
plants_2023.loc[X.index, 'pca2'] = X_pca[:, 1]

print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
print(f"Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ======================================================================
# Code Block 9
# ======================================================================

from sklearn.manifold import TSNE

# t-SNE on sample (slow for large data)
sample_idx = np.random.choice(len(X_scaled), 2000, replace=False)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled[sample_idx])

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
           cmap='tab10', alpha=0.6, s=20)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
           c=cluster_labels[sample_idx], 
           cmap='tab10', alpha=0.6, s=20)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Projection (sample)')
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=150)

# ======================================================================
# Code Block 10
# ======================================================================

# For each plant, compare to cluster peers
def benchmark_plant(plant_row, cluster_data):
    cluster_median = cluster_data['carbon_intensity'].median()
    plant_value = plant_row['carbon_intensity']
    
    percentile = (cluster_data['carbon_intensity'] < plant_value).mean() * 100
    
    return {
        'cluster_median': cluster_median,
        'plant_value': plant_value,
        'percentile': percentile,
        'vs_median': ((plant_value - cluster_median) / cluster_median * 100)
    }

# Example: Benchmark a specific coal plant
coal_cluster_id = 0  # From profiling above
coal_plants = plants_2023[plants_2023['cluster'] == coal_cluster_id]

example_plant = coal_plants.sample(1).iloc[0]
benchmark = benchmark_plant(example_plant, coal_plants)

print(f"Plant: {example_plant.get('Plant name', 'Unknown')}")
print(f"Carbon intensity: {benchmark['plant_value']:.3f} tons/MWh")
print(f"Cluster median: {benchmark['cluster_median']:.3f} tons/MWh")
print(f"Performance: {benchmark['vs_median']:+.1f}% vs peers")
print(f"Percentile: {benchmark['percentile']:.0f}th")

# ======================================================================
# Code Block 11
# ======================================================================

# Aggregate to state level
state_profiles = plants_2023.groupby('Plant state abbreviation').apply(
    lambda df: pd.Series({
        'pct_cluster_0': (df['cluster'] == 0).sum() / len(df) * 100,
        'pct_cluster_1': (df['cluster'] == 1).sum() / len(df) * 100,
        'pct_cluster_2': (df['cluster'] == 2).sum() / len(df) * 100,
        'pct_cluster_3': (df['cluster'] == 3).sum() / len(df) * 100,
        'pct_cluster_4': (df['cluster'] == 4).sum() / len(df) * 100,
    })
)

# Cluster states
state_scaler = StandardScaler()
state_profiles_scaled = state_scaler.fit_transform(state_profiles)

state_kmeans = KMeans(n_clusters=4, random_state=42)
state_clusters = state_kmeans.fit_predict(state_profiles_scaled)

state_profiles['state_cluster'] = state_clusters

print("\nState Cluster Profiles:")
for i in range(4):
    states_in_cluster = state_profiles[state_profiles['state_cluster'] == i].index.tolist()
    print(f"\nCluster {i}: {', '.join(states_in_cluster)}")
    print(state_profiles[state_profiles['state_cluster'] == i].mean())

# ======================================================================
# Code Block 12
# ======================================================================

# Find similar plants for a target plant
from sklearn.neighbors import NearestNeighbors

def find_similar_plants(target_plant_idx, X_scaled, n_neighbors=10):
    nn = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn.fit(X_scaled)
    
    distances, indices = nn.kneighbors([X_scaled[target_plant_idx]])
    
    # Exclude the target plant itself (distance=0)
    return indices[0][1:], distances[0][1:]

# Example: Find competitors for a specific plant
target_idx = 1000
similar_idx, similar_dist = find_similar_plants(target_idx, X_scaled)

print("Most similar plants (competitors):")
for i, (idx, dist) in enumerate(zip(similar_idx, similar_dist), 1):
    plant = plants_2023.iloc[X.index[idx]]
    print(f"{i}. {plant.get('Plant name', 'Unknown')} "
          f"({plant.get('Plant state abbreviation', '??')}) - "
          f"Distance: {dist:.3f}")

# ======================================================================
# Code Block 13
# ======================================================================

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

results.append({
    'k': k,
    'inertia': kmeans.inertia_,
    'silhouette': silhouette_score(X_scaled, labels),
    'calinski': calinski_harabasz_score(X_scaled, labels)
})
print(f"K={k}: Silhouette={results[-1]['silhouette']:.3f}")

# ======================================================================
# Code Block 14
# ======================================================================

K=2: Silhouette=0.412
K=3: Silhouette=0.389
K=4: Silhouette=0.367
K=5: Silhouette=0.356  <- Sweet spot
K=6: Silhouette=0.341
K=7: Silhouette=0.329

# ======================================================================
# Code Block 15
# ======================================================================

gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
gmm.fit(X_scaled)
bic_scores.append(gmm.bic(X_scaled))
print(f"n={n}: BIC={gmm.bic(X_scaled):.1f}")

# ======================================================================
# Code Block 16
# ======================================================================

n_components=5, 
covariance_type='full',
random_state=42

# ======================================================================
# Code Block 17
# ======================================================================

min_cluster_size=50,  # Minimum 50 plants per cluster
min_samples=10,
metric='euclidean'

# ======================================================================
# Code Block 18
# ======================================================================

cmap='tab10', alpha=0.6, s=20)

# ======================================================================
# Code Block 19
# ======================================================================

c=cluster_labels[sample_idx], 
       cmap='tab10', alpha=0.6, s=20)

# ======================================================================
# Code Block 20
# ======================================================================

cluster_median = cluster_data['carbon_intensity'].median()
plant_value = plant_row['carbon_intensity']

percentile = (cluster_data['carbon_intensity'] < plant_value).mean() * 100

return {
    'cluster_median': cluster_median,
    'plant_value': plant_value,
    'percentile': percentile,
    'vs_median': ((plant_value - cluster_median) / cluster_median * 100)
}

# ======================================================================
# Code Block 21
# ======================================================================

lambda df: pd.Series({
    'pct_cluster_0': (df['cluster'] == 0).sum() / len(df) * 100,
    'pct_cluster_1': (df['cluster'] == 1).sum() / len(df) * 100,
    'pct_cluster_2': (df['cluster'] == 2).sum() / len(df) * 100,
    'pct_cluster_3': (df['cluster'] == 3).sum() / len(df) * 100,
    'pct_cluster_4': (df['cluster'] == 4).sum() / len(df) * 100,
})

# ======================================================================
# Code Block 22
# ======================================================================

states_in_cluster = state_profiles[state_profiles['state_cluster'] == i].index.tolist()
print(f"\nCluster {i}: {', '.join(states_in_cluster)}")
print(state_profiles[state_profiles['state_cluster'] == i].mean())

# ======================================================================
# Code Block 23
# ======================================================================

nn = NearestNeighbors(n_neighbors=n_neighbors+1)
nn.fit(X_scaled)

distances, indices = nn.kneighbors([X_scaled[target_plant_idx]])

# ======================================================================
# Code Block 24
# ======================================================================

return indices[0][1:], distances[0][1:]

# ======================================================================
# Code Block 25
# ======================================================================

plant = plants_2023.iloc[X.index[idx]]
print(f"{i}. {plant.get('Plant name', 'Unknown')} "
      f"({plant.get('Plant state abbreviation', '??')}) - "
      f"Distance: {dist:.3f}")
