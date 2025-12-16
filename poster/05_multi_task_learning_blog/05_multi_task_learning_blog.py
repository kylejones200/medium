#!/usr/bin/env python3
"""
Python code extracted from 05_multi_task_learning_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
plants = pd.read_parquet('egrid_all_plants_1996-2023.parquet')
plants_2023 = plants[plants['data_year'] == 2023].copy()

# Get emissions columns
co2 = pd.to_numeric(plants_2023['Plant annual CO2 emissions (tons)'], errors='coerce')
nox = pd.to_numeric(plants_2023['Plant annual NOx emissions (tons)'], errors='coerce')
so2 = pd.to_numeric(plants_2023['Plant annual SO2 emissions (tons)'], errors='coerce')

# Log transform (emissions are heavily skewed)
emissions_df = pd.DataFrame({
    'log_co2': np.log1p(co2),
    'log_nox': np.log1p(nox),
    'log_so2': np.log1p(so2)
}).dropna()

# Correlation matrix
corr = emissions_df.corr()
print("Emissions Correlations:")
print(corr)

# Visualize
sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, 
           square=True, linewidths=2, cbar_kws={"shrink": 0.8})
plt.title('Pollutant Correlations: Why MTL Works')
plt.tight_layout()
plt.savefig('pollutant_correlations.png', dpi=150)

# ======================================================================
# Code Block 2
# ======================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_mtl_model(input_dim, architecture='hard_sharing'):
    """
    Build multi-task learning model
    
    Parameters:
    - input_dim: Number of input features
    - architecture: 'hard_sharing' or 'soft_sharing'
    
    Returns:
    - Keras model with three outputs (CO2, NOx, SO2)
    """
    
    # Input layer
    inputs = keras.Input(shape=(input_dim,), name='input_features')
    
    # Shared layers - learn common patterns
    shared = layers.Dense(128, activation='relu', name='shared_1')(inputs)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    shared = layers.Dense(64, activation='relu', name='shared_2')(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    shared = layers.Dense(32, activation='relu', name='shared_3')(shared)
    shared = layers.BatchNormalization()(shared)
    
    # Task-specific heads
    # CO2 head
    co2_head = layers.Dense(16, activation='relu', name='co2_head')(shared)
    co2_output = layers.Dense(1, name='co2_output')(co2_head)
    
    # NOx head
    nox_head = layers.Dense(16, activation='relu', name='nox_head')(shared)
    nox_output = layers.Dense(1, name='nox_output')(nox_head)
    
    # SO2 head
    so2_head = layers.Dense(16, activation='relu', name='so2_head')(shared)
    so2_output = layers.Dense(1, name='so2_output')(so2_head)
    
    # Build model
    model = keras.Model(
        inputs=inputs,
        outputs=[co2_output, nox_output, so2_output],
        name='mtl_emissions_predictor'
    )
    
    return model

# Create model
model = build_mtl_model(input_dim=10)
model.summary()

# ======================================================================
# Code Block 3
# ======================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features: plant characteristics and operating conditions
feature_cols = [
    'Plant nameplate capacity (MW)',
    'Plant annual net generation (MWh)',
    'Plant annual heat input (MMBtu)',
    'Plant state abbreviation',  # One-hot encode
    'Plant primary fuel category',  # One-hot encode
]

# Prepare features
X = plants_2023[feature_cols].copy()

# Handle categorical variables
X['capacity_mw'] = pd.to_numeric(X['Plant nameplate capacity (MW)'], errors='coerce')
X['generation_mwh'] = pd.to_numeric(X['Plant annual net generation (MWh)'], errors='coerce')
X['heat_input_mmbtu'] = pd.to_numeric(X['Plant annual heat input (MMBtu)'], errors='coerce')

# One-hot encode state and fuel type
X_encoded = pd.get_dummies(X[['Plant state abbreviation', 'Plant primary fuel category']], 
                           drop_first=True)

# Combine
X_features = pd.concat([
    X[['capacity_mw', 'generation_mwh', 'heat_input_mmbtu']],
    X_encoded
], axis=1)

# Targets
y_co2 = emissions_df['log_co2']
y_nox = emissions_df['log_nox']
y_so2 = emissions_df['log_so2']

# Align indices (only plants with all data)
common_idx = X_features.index.intersection(y_co2.index)
X_features = X_features.loc[common_idx]
y_co2 = y_co2.loc[common_idx]
y_nox = y_nox.loc[common_idx]
y_so2 = y_so2.loc[common_idx]

print(f"Training on {len(X_features):,} plants")

# Train/test split
X_train, X_test, y_co2_train, y_co2_test = train_test_split(
    X_features, y_co2, test_size=0.2, random_state=42
)

_, _, y_nox_train, y_nox_test = train_test_split(
    X_features, y_nox, test_size=0.2, random_state=42
)

_, _, y_so2_train, y_so2_test = train_test_split(
    X_features, y_so2, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# ======================================================================
# Code Block 4
# ======================================================================

# Rebuild model with correct input dimension
input_dim = X_train_scaled.shape[1]
mtl_model = build_mtl_model(input_dim)

# Compile with multiple losses
mtl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'co2_output': 'mse',
        'nox_output': 'mse',
        'so2_output': 'mse'
    },
    metrics={
        'co2_output': ['mae', 'mse'],
        'nox_output': ['mae', 'mse'],
        'so2_output': ['mae', 'mse']
    }
)

# Train
history = mtl_model.fit(
    X_train_scaled,
    {
        'co2_output': y_co2_train,
        'nox_output': y_nox_train,
        'so2_output': y_so2_train
    },
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ],
    verbose=1
)

# Evaluate
results = mtl_model.evaluate(
    X_test_scaled,
    {
        'co2_output': y_co2_test,
        'nox_output': y_nox_test,
        'so2_output': y_so2_test
    }
)

print("\nMTL Model Performance:")
print(f"CO2 MAE: {results[4]:.4f}")
print(f"NOx MAE: {results[7]:.4f}")
print(f"SO2 MAE: {results[10]:.4f}")

# ======================================================================
# Code Block 5
# ======================================================================

def build_single_task_model(input_dim):
    """Single-task model for comparison"""
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

# Train three separate models
print("\nTraining single-task baselines...")

# CO2 model
co2_model = build_single_task_model(input_dim)
co2_model.fit(X_train_scaled, y_co2_train, 
             validation_split=0.2, epochs=50, batch_size=64,
             callbacks=[keras.callbacks.EarlyStopping(patience=10)],
             verbose=0)
co2_baseline_mae = co2_model.evaluate(X_test_scaled, y_co2_test, verbose=0)[1]

# NOx model
nox_model = build_single_task_model(input_dim)
nox_model.fit(X_train_scaled, y_nox_train, 
             validation_split=0.2, epochs=50, batch_size=64,
             callbacks=[keras.callbacks.EarlyStopping(patience=10)],
             verbose=0)
nox_baseline_mae = nox_model.evaluate(X_test_scaled, y_nox_test, verbose=0)[1]

# SO2 model
so2_model = build_single_task_model(input_dim)
so2_model.fit(X_train_scaled, y_so2_train, 
             validation_split=0.2, epochs=50, batch_size=64,
             callbacks=[keras.callbacks.EarlyStopping(patience=10)],
             verbose=0)
so2_baseline_mae = so2_model.evaluate(X_test_scaled, y_so2_test, verbose=0)[1]

print("\nSingle-Task Baselines:")
print(f"CO2 MAE: {co2_baseline_mae:.4f}")
print(f"NOx MAE: {nox_baseline_mae:.4f}")
print(f"SO2 MAE: {so2_baseline_mae:.4f}")

# ======================================================================
# Code Block 6
# ======================================================================

# Compare
comparison = pd.DataFrame({
    'Task': ['CO₂', 'NOx', 'SO₂'],
    'Single-Task MAE': [co2_baseline_mae, nox_baseline_mae, so2_baseline_mae],
    'MTL MAE': [results[4], results[7], results[10]]
})

comparison['Improvement %'] = (
    (comparison['Single-Task MAE'] - comparison['MTL MAE']) / 
    comparison['Single-Task MAE'] * 100
)

print("\nMTL vs Single-Task Comparison:")
print(comparison.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison))
width = 0.35

ax.bar(x - width/2, comparison['Single-Task MAE'], width, 
       label='Single-Task', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, comparison['MTL MAE'], width, 
       label='Multi-Task', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Pollutant', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (log scale)', fontsize=12, fontweight='bold')
ax.set_title('Multi-Task Learning Improves All Tasks', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Task'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add improvement annotations
for i, row in comparison.iterrows():
    ax.text(i, max(row['Single-Task MAE'], row['MTL MAE']) + 0.02,
           f'+{row["Improvement %"]:.1f}%',
           ha='center', fontsize=11, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('mtl_vs_single_task.png', dpi=150)

# ======================================================================
# Code Block 7
# ======================================================================

# Rebuild with weighted losses
mtl_model_weighted = build_mtl_model(input_dim)

mtl_model_weighted.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'co2_output': 'mse',
        'nox_output': 'mse',
        'so2_output': 'mse'
    },
    loss_weights={
        'co2_output': 2.0,  # CO2 is more important
        'nox_output': 1.0,
        'so2_output': 1.0
    },
    metrics={
        'co2_output': ['mae'],
        'nox_output': ['mae'],
        'so2_output': ['mae']
    }
)

# Train
history_weighted = mtl_model_weighted.fit(
    X_train_scaled,
    {
        'co2_output': y_co2_train,
        'nox_output': y_nox_train,
        'so2_output': y_so2_train
    },
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
    verbose=0
)

# Evaluate
results_weighted = mtl_model_weighted.evaluate(
    X_test_scaled,
    {
        'co2_output': y_co2_test,
        'nox_output': y_nox_test,
        'so2_output': y_so2_test
    },
    verbose=0
)

print("\nWeighted MTL (CO2 priority):")
print(f"CO2 MAE: {results_weighted[2]:.4f} (improved!)")
print(f"NOx MAE: {results_weighted[3]:.4f} (slightly worse)")
print(f"SO2 MAE: {results_weighted[4]:.4f} (slightly worse)")

# ======================================================================
# Code Block 8
# ======================================================================

# Compare MTL to single-task
if mtl_mae > single_task_mae:
    print("⚠️ Negative transfer detected!")
    print("Tasks may be unrelated or need different architectures")

# ======================================================================
# Code Block 9
# ======================================================================

def build_soft_sharing_model(input_dim, n_tasks=3):
    """
    Soft parameter sharing: separate networks with L2 regularization
    encouraging similarity
    """
    inputs = keras.Input(shape=(input_dim,))
    
    # Separate networks for each task
    task_networks = []
    for i in range(n_tasks):
        net = layers.Dense(128, activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
        net = layers.Dense(64, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.01))(net)
        net = layers.Dense(32, activation='relu')(net)
        task_networks.append(net)
    
    # Task-specific outputs
    co2_output = layers.Dense(1, name='co2_output')(task_networks[0])
    nox_output = layers.Dense(1, name='nox_output')(task_networks[1])
    so2_output = layers.Dense(1, name='so2_output')(task_networks[2])
    
    model = keras.Model(inputs=inputs, 
                       outputs=[co2_output, nox_output, so2_output])
    
    return model

# ======================================================================
# Code Block 10
# ======================================================================

# Save model
mtl_model.save('mtl_emissions_predictor.h5')

# Load and predict
loaded_model = keras.models.load_model('mtl_emissions_predictor.h5')

# New plant data
new_plant = X_test_scaled[:1]  # Example

# Predict all three pollutants in one call
co2_pred, nox_pred, so2_pred = loaded_model.predict(new_plant)

print(f"Predicted CO2: {np.expm1(co2_pred[0][0]):,.0f} tons")
print(f"Predicted NOx: {np.expm1(nox_pred[0][0]):,.0f} tons")
print(f"Predicted SO2: {np.expm1(so2_pred[0][0]):,.0f} tons")

# ======================================================================
# Code Block 11
# ======================================================================

square=True, linewidths=2, cbar_kws={"shrink": 0.8})

# ======================================================================
# Code Block 12
# ======================================================================

inputs = keras.Input(shape=(input_dim,), name='input_features')

# ======================================================================
# Code Block 13
# ======================================================================

shared = layers.Dense(128, activation='relu', name='shared_1')(inputs)
shared = layers.BatchNormalization()(shared)
shared = layers.Dropout(0.3)(shared)

shared = layers.Dense(64, activation='relu', name='shared_2')(shared)
shared = layers.BatchNormalization()(shared)
shared = layers.Dropout(0.3)(shared)

shared = layers.Dense(32, activation='relu', name='shared_3')(shared)
shared = layers.BatchNormalization()(shared)

# ======================================================================
# Code Block 14
# ======================================================================

co2_head = layers.Dense(16, activation='relu', name='co2_head')(shared)
co2_output = layers.Dense(1, name='co2_output')(co2_head)

# ======================================================================
# Code Block 15
# ======================================================================

nox_head = layers.Dense(16, activation='relu', name='nox_head')(shared)
nox_output = layers.Dense(1, name='nox_output')(nox_head)

# ======================================================================
# Code Block 16
# ======================================================================

so2_head = layers.Dense(16, activation='relu', name='so2_head')(shared)
so2_output = layers.Dense(1, name='so2_output')(so2_head)

# ======================================================================
# Code Block 17
# ======================================================================

model = keras.Model(
    inputs=inputs,
    outputs=[co2_output, nox_output, so2_output],
    name='mtl_emissions_predictor'
)

return model

# ======================================================================
# Code Block 18
# ======================================================================

Model: "mtl_emissions_predictor"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_features (InputLayer)    [(None, 10)]         0           []                               
                                                                                                  
 shared_1 (Dense)               (None, 128)          1408        ['input_features[0][0]']         
                                                                                                  
 batch_normalization (BatchNorm (None, 128)          512         ['shared_1[0][0]']               
                                                                                                  
 dropout (Dropout)              (None, 128)          0           ['batch_normalization[0][0]']    
                                                                                                  
 shared_2 (Dense)               (None, 64)           8256        ['dropout[0][0]']                
                                                                                                  
 batch_normalization_1 (BatchNo (None, 64)           256         ['shared_2[0][0]']               
                                                                                                  
 dropout_1 (Dropout)            (None, 64)           0           ['batch_normalization_1[0][0]']  
                                                                                                  
 shared_3 (Dense)               (None, 32)           2080        ['dropout_1[0][0]']              
                                                                                                  
 batch_normalization_2 (BatchNo (None, 32)           128         ['shared_3[0][0]']               
                                                                                                  
 co2_head (Dense)               (None, 16)           528         ['batch_normalization_2[0][0]']  
                                                                                                  
 nox_head (Dense)               (None, 16)           528         ['batch_normalization_2[0][0]']  
                                                                                                  
 so2_head (Dense)               (None, 16)           528         ['batch_normalization_2[0][0]']  
                                                                                                  
 co2_output (Dense)             (None, 1)            17          ['co2_head[0][0]']               
                                                                                                  
 nox_output (Dense)             (None, 1)            17          ['nox_head[0][0]']               
                                                                                                  
 so2_output (Dense)             (None, 1)            17          ['so2_head[0][0]']               
==================================================================================================
Total params: 14,275
Trainable params: 13,827
Non-trainable params: 448
__________________________________________________________________________________________________

# ======================================================================
# Code Block 19
# ======================================================================

drop_first=True)

# ======================================================================
# Code Block 20
# ======================================================================

X_features, y_co2, test_size=0.2, random_state=42

# ======================================================================
# Code Block 21
# ======================================================================

X_features, y_nox, test_size=0.2, random_state=42

# ======================================================================
# Code Block 22
# ======================================================================

X_features, y_so2, test_size=0.2, random_state=42

# ======================================================================
# Code Block 23
# ======================================================================

optimizer=keras.optimizers.Adam(learning_rate=0.001),
loss={
    'co2_output': 'mse',
    'nox_output': 'mse',
    'so2_output': 'mse'
},
metrics={
    'co2_output': ['mae', 'mse'],
    'nox_output': ['mae', 'mse'],
    'so2_output': ['mae', 'mse']
}

# ======================================================================
# Code Block 24
# ======================================================================

X_train_scaled,
{
    'co2_output': y_co2_train,
    'nox_output': y_nox_train,
    'so2_output': y_so2_train
},
validation_split=0.2,
epochs=50,
batch_size=64,
callbacks=[
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
],
verbose=1

# ======================================================================
# Code Block 25
# ======================================================================

"""Single-task model for comparison"""
model = keras.Sequential([
    keras.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

return model

# ======================================================================
# Code Block 26
# ======================================================================

validation_split=0.2, epochs=50, batch_size=64,
         callbacks=[keras.callbacks.EarlyStopping(patience=10)],
         verbose=0)

# ======================================================================
# Code Block 27
# ======================================================================

validation_split=0.2, epochs=50, batch_size=64,
         callbacks=[keras.callbacks.EarlyStopping(patience=10)],
         verbose=0)

# ======================================================================
# Code Block 28
# ======================================================================

validation_split=0.2, epochs=50, batch_size=64,
         callbacks=[keras.callbacks.EarlyStopping(patience=10)],
         verbose=0)

# ======================================================================
# Code Block 29
# ======================================================================

label='Single-Task', color='#e74c3c', alpha=0.8)

# ======================================================================
# Code Block 30
# ======================================================================

label='Multi-Task', color='#2ecc71', alpha=0.8)

# ======================================================================
# Code Block 31
# ======================================================================

ax.text(i, max(row['Single-Task MAE'], row['MTL MAE']) + 0.02,
       f'+{row["Improvement %"]:.1f}%',
       ha='center', fontsize=11, fontweight='bold', color='green')

# ======================================================================
# Code Block 32
# ======================================================================

optimizer=keras.optimizers.Adam(learning_rate=0.001),
loss={
    'co2_output': 'mse',
    'nox_output': 'mse',
    'so2_output': 'mse'
},
loss_weights={
    'co2_output': 2.0,  # CO2 is more important
    'nox_output': 1.0,
    'so2_output': 1.0
},
metrics={
    'co2_output': ['mae'],
    'nox_output': ['mae'],
    'so2_output': ['mae']
}

# ======================================================================
# Code Block 33
# ======================================================================

X_train_scaled,
{
    'co2_output': y_co2_train,
    'nox_output': y_nox_train,
    'so2_output': y_so2_train
},
validation_split=0.2,
epochs=50,
batch_size=64,
callbacks=[keras.callbacks.EarlyStopping(patience=10)],
verbose=0

# ======================================================================
# Code Block 34
# ======================================================================

X_test_scaled,
{
    'co2_output': y_co2_test,
    'nox_output': y_nox_test,
    'so2_output': y_so2_test
},
verbose=0

# ======================================================================
# Code Block 35
# ======================================================================

print("⚠️ Negative transfer detected!")
print("Tasks may be unrelated or need different architectures")

# ======================================================================
# Code Block 36
# ======================================================================

"""
Soft parameter sharing: separate networks with L2 regularization
encouraging similarity
"""
inputs = keras.Input(shape=(input_dim,))

# ======================================================================
# Code Block 37
# ======================================================================

task_networks = []
for i in range(n_tasks):
    net = layers.Dense(128, activation='relu', 
                      kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
    net = layers.Dense(64, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(0.01))(net)
    net = layers.Dense(32, activation='relu')(net)
    task_networks.append(net)

# ======================================================================
# Code Block 38
# ======================================================================

co2_output = layers.Dense(1, name='co2_output')(task_networks[0])
nox_output = layers.Dense(1, name='nox_output')(task_networks[1])
so2_output = layers.Dense(1, name='so2_output')(task_networks[2])

model = keras.Model(inputs=inputs, 
                   outputs=[co2_output, nox_output, so2_output])

return model
