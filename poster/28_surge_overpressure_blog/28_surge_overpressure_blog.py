#!/usr/bin/env python3
"""
Python code extracted from 28_surge_overpressure_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
import pandas as pd

def generate_surge_scenarios(n_scenarios=5000, seed=2025):
    """
    Generate synthetic transient scenarios with physics-inspired relationships.
    """
    rng = np.random.default_rng(seed)
    
    # Input parameters (varied across scenarios)
    linepack = rng.uniform(0.6, 1.4, n_scenarios)         # Relative to normal (dimensionless)
    closure_time = rng.uniform(0.2, 10.0, n_scenarios)    # Seconds
    pump_trip = rng.integers(0, 2, n_scenarios)           # 0=no, 1=yes
    velocity = rng.uniform(0.5, 3.0, n_scenarios)         # m/s
    elevation_drop = rng.uniform(-80, 120, n_scenarios)   # m (negative = uphill)
    temperature = rng.uniform(0, 35, n_scenarios)         # °C
    
    # Physics-based target: peak overpressure
    # Components:
    # 1. Joukowsky surge (inversely proportional to closure time)
    base_surge = 35 * velocity / (1 + closure_time / 2.0)
    
    # 2. Static head contribution (elevation changes)
    static_head = 0.433 * (elevation_drop / 10.0)  # psi per 10m
    
    # 3. Pump trip amplification
    pump_effect = pump_trip * (12 + 6 * np.tanh(3 * (1.5 - velocity)))
    
    # 4. Linepack (compressibility) effect
    linepack_effect = 8 * (linepack - 1.0)
    
    # 5. Temperature effect (minor, via fluid properties)
    temp_effect = 0.2 * temperature
    
    # 6. Realistic noise
    noise = rng.normal(0, 2.0, n_scenarios)
    
    # Baseline operating pressure + surge components
    peak_overpress = (
        200 +  # Baseline operating pressure (psig)
        base_surge +
        static_head +
        pump_effect +
        linepack_effect +
        temp_effect +
        noise
    )
    
    df = pd.DataFrame({
        'linepack': linepack,
        'closure_time_s': closure_time,
        'pump_trip': pump_trip,
        'velocity_ms': velocity,
        'elevation_drop_m': elevation_drop,
        'temperature_c': temperature,
        'peak_overpress_psig': peak_overpress
    })
    
    return df

# Generate training data
df_train = generate_surge_scenarios(n_scenarios=5000)
print(f'Generated {len(df_train):,} scenarios')
print(f'Peak overpressure range: {df_train["peak_overpress_psig"].min():.1f} - {df_train["peak_overpress_psig"].max():.1f} psig')

# ======================================================================
# Code Block 2
# ======================================================================

# Test: Joukowsky equation for instant closure
# Expected: ΔP ≈ ρ × a × Δv / 145 (convert Pa to psi)
# For oil: ρ=850 kg/m³, a=1200 m/s, v=2.5 m/s
# ΔP = 850 * 1200 * 2.5 / 6895 ≈ 370 psi

test_instant = df_train[
    (df_train['closure_time_s'] < 0.5) &
    (df_train['velocity_ms'] > 2.4) &
    (df_train['pump_trip'] == 0) &
    (df_train['linepack'].between(0.95, 1.05))
]

instant_surge = test_instant['peak_overpress_psig'] - 200  # Remove baseline
print(f'Instant closure surge: {instant_surge.mean():.1f} ± {instant_surge.std():.1f} psi')
# Expected output: ~87 psi (matches 35*2.5 from base_surge formula)

# ======================================================================
# Code Block 3
# ======================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Features and target
X = df_train.drop(columns=['peak_overpress_psig'])
y = df_train['peak_overpress_psig']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Pipeline: Scaling + Gradient Boosting
model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', HistGradientBoostingRegressor(
        max_depth=5,
        max_iter=500,
        learning_rate=0.05,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Test MAE: {mae:.2f} psi')
print(f'Test R²: {r2:.4f}')

# ======================================================================
# Code Block 4
# ======================================================================

import matplotlib.pyplot as plt

# Extract feature importance
feature_names = X.columns.tolist()
importances = model.named_steps['gbr'].feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Feature Importance')
plt.title('Gradient Boosting Feature Importance for Surge Prediction')
plt.tight_layout()
plt.show()

# ======================================================================
# Code Block 5
# ======================================================================

def predict_safe_closure_time(
    velocity_ms,
    linepack=1.0,
    pump_trip=0,
    elevation_drop_m=0,
    temperature_c=15,
    max_allowable_pressure=260
):
    """
    Find minimum closure time that keeps peak pressure below MAOP.
    """
    closure_times = np.linspace(0.2, 15, 150)
    
    scenarios = pd.DataFrame({
        'linepack': linepack,
        'closure_time_s': closure_times,
        'pump_trip': pump_trip,
        'velocity_ms': velocity_ms,
        'elevation_drop_m': elevation_drop_m,
        'temperature_c': temperature_c
    })
    
    predicted_peaks = model.predict(scenarios)
    
    # Find minimum closure time where peak < MAOP
    safe_indices = np.where(predicted_peaks < max_allowable_pressure)[0]
    
    if len(safe_indices) == 0:
        return None, predicted_peaks  # No safe closure time exists
    
    min_safe_time = closure_times[safe_indices[0]]
    return min_safe_time, predicted_peaks

# Example: Calculate safe closure for current operating conditions
velocity = 2.2  # m/s (from SCADA)
safe_time, all_peaks = predict_safe_closure_time(
    velocity_ms=velocity,
    linepack=1.1,  # Slightly overpacked
    pump_trip=0,
    max_allowable_pressure=260
)

print(f'Current velocity: {velocity} m/s')
print(f'Minimum safe closure time: {safe_time:.1f} seconds')
print(f'Peak pressure at safe time: {all_peaks[int(safe_time*10)]:.1f} psig')

# ======================================================================
# Code Block 6
# ======================================================================

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(10, 6))

# Generate surge curves for low, medium, high velocity
velocities = [0.8, 1.5, 2.5]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
closure_grid = np.linspace(0.2, 12, 120)

for v, color, label in zip(velocities, colors, ['Low (0.8 m/s)', 'Medium (1.5 m/s)', 'High (2.5 m/s)']):
    probe = pd.DataFrame({
        'linepack': 1.0,
        'closure_time_s': closure_grid,
        'pump_trip': 0,
        'velocity_ms': v,
        'elevation_drop_m': 0,
        'temperature_c': 15
    })
    
    predicted = model.predict(probe)
    ax.plot(closure_grid, predicted, color=color, linewidth=2.5, label=label)

# Add MAOP line
ax.axhline(y=260, color='red', linestyle='--', linewidth=2, label='MAOP (260 psig)')

# Add "safe zone" shading
ax.fill_between(closure_grid, 0, 260, alpha=0.1, color='green', label='Safe Zone')

ax.set_xlabel('Valve Closure Time (seconds)', fontsize=12)
ax.set_ylabel('Predicted Peak Overpressure (psig)', fontsize=12)
ax.set_title('Surge vs. Closure Time by Flow Velocity', fontsize=14, pad=15)
ax.legend(loc='upper right', frameon=False, fontsize=10)
ax.grid(True, alpha=0.3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))

plt.tight_layout()
plt.savefig('surge_vs_closure_time.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# Code Block 7
# ======================================================================

# Real-time SCADA monitoring
def detect_pump_trip(pressure_trace, flow_trace, window=10):
    """
    Detect sudden pump loss from pressure/flow signatures.
    """
    # Check for simultaneous pressure drop + flow drop
    dpdt = np.diff(pressure_trace[-window:]).mean()
    dqdt = np.diff(flow_trace[-window:]).mean()
    
    if dpdt < -5 and dqdt < -0.2:  # Thresholds from historical data
        return True
    return False

# If pump trip detected, use longer closure time to avoid surge amplification
if detect_pump_trip(recent_pressure, recent_flow):
    safe_closure_time = predict_safe_closure_time(
        velocity_ms=current_velocity,
        pump_trip=1,  # Flag trip in model
        max_allowable_pressure=260
    )
    print(f'PUMP TRIP DETECTED: Use {safe_closure_time:.1f}s closure (extended)')

# ======================================================================
# Code Block 8
# ======================================================================

# Model surge propagation through network
# Features: valve location, upstream/downstream segments, boundary conditions
# Output: peak pressure at each valve location

# Use graph neural network (GNN) to capture network topology
from torch_geometric.nn import GCNConv

class PipelineNetworkSurge(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)  # Output: pressure at each node
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ======================================================================
# Code Block 9
# ======================================================================

from sklearn.ensemble import GradientBoostingRegressor

# Train ensemble of models with bootstrap sampling
n_estimators = 50
predictions = []

for i in range(n_estimators):
    # Bootstrap sample
    sample_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train.iloc[sample_idx]
    y_boot = y_train.iloc[sample_idx]
    
    # Train model
    model_i = GradientBoostingRegressor(random_state=i)
    model_i.fit(X_boot, y_boot)
    
    # Predict
    pred_i = model_i.predict(X_test)
    predictions.append(pred_i)

# Compute mean and confidence intervals
predictions = np.array(predictions)
mean_pred = predictions.mean(axis=0)
lower_bound = np.percentile(predictions, 2.5, axis=0)
upper_bound = np.percentile(predictions, 97.5, axis=0)

print(f'Prediction: {mean_pred[0]:.1f} psig [95% CI: {lower_bound[0]:.1f} - {upper_bound[0]:.1f}]')

# ======================================================================
# Code Block 10
# ======================================================================

# When actual transient event occurs, record data
def record_transient_event(scada_data, peak_pressure_measured):
    """
    Log real event for model retraining.
    """
    event_features = {
        'linepack': scada_data['linepack'],
        'closure_time_s': scada_data['closure_time'],
        'pump_trip': scada_data['pump_trip'],
        'velocity_ms': scada_data['velocity'],
        'elevation_drop_m': scada_data['elevation'],
        'temperature_c': scada_data['temperature'],
        'peak_overpress_psig': peak_pressure_measured
    }
    
    # Append to training database
    historical_events.append(event_features)
    
    # Trigger model retrain if 20+ new events collected
    if len(historical_events) >= 20:
        retrain_model(historical_events)

# ======================================================================
# Code Block 11
# ======================================================================

# Complete surge surrogate modeling pipeline

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================================================================
# 1. Generate Training Data
# ============================================================================

def generate_surge_scenarios(n=5000, seed=2025):
    rng = np.random.default_rng(seed)
    
    linepack = rng.uniform(0.6, 1.4, n)
    closure_time = rng.uniform(0.2, 10.0, n)
    pump_trip = rng.integers(0, 2, n)
    velocity = rng.uniform(0.5, 3.0, n)
    elev = rng.uniform(-80, 120, n)
    temp = rng.uniform(0, 35, n)
    
    base = 35 * velocity / (1 + closure_time / 2.0)
    head = 0.433 * (elev / 10.0)
    trip = pump_trip * (12 + 6 * np.tanh(3 * (1.5 - velocity)))
    pack = 8 * (linepack - 1.0)
    noise = rng.normal(0, 2.0, n)
    
    peak = 200 + base + head + trip + pack + 0.2 * temp + noise
    
    return pd.DataFrame({
        'linepack': linepack,
        'closure_time_s': closure_time,
        'pump_trip': pump_trip,
        'velocity_ms': velocity,
        'elevation_drop_m': elev,
        'temperature_c': temp,
        'peak_overpress_psig': peak
    })

df = generate_surge_scenarios(5000)
print(f'✓ Generated {len(df):,} scenarios')

# ============================================================================
# 2. Train Model
# ============================================================================

X = df.drop(columns=['peak_overpress_psig'])
y = df['peak_overpress_psig']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', HistGradientBoostingRegressor(max_depth=5, max_iter=500, learning_rate=0.05, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'✓ Model trained: MAE={mae:.2f} psi, R²={r2:.4f}')

# ============================================================================
# 3. Safe Closure Time Calculator
# ============================================================================

def predict_safe_closure(velocity, linepack=1.0, pump_trip=0, elev=0, temp=15, maop=260):
    times = np.linspace(0.2, 15, 150)
    scenarios = pd.DataFrame({
        'linepack': linepack, 'closure_time_s': times, 'pump_trip': pump_trip,
        'velocity_ms': velocity, 'elevation_drop_m': elev, 'temperature_c': temp
    })
    peaks = model.predict(scenarios)
    safe = np.where(peaks < maop)[0]
    return times[safe[0]] if len(safe) > 0 else None, peaks

# Example
v = 2.2
safe_time, _ = predict_safe_closure(v)
print(f'✓ Velocity={v} m/s → Safe closure time: {safe_time:.1f}s')

# ============================================================================
# 4. Visualization
# ============================================================================

plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(10, 6))

closure_grid = np.linspace(0.2, 12, 120)
for v, color, label in [(0.8, '#2ecc71', 'Low (0.8 m/s)'),
                         (1.5, '#f39c12', 'Medium (1.5 m/s)'),
                         (2.5, '#e74c3c', 'High (2.5 m/s)')]:
    probe = pd.DataFrame({
        'linepack': 1.0, 'closure_time_s': closure_grid, 'pump_trip': 0,
        'velocity_ms': v, 'elevation_drop_m': 0, 'temperature_c': 15
    })
    pred = model.predict(probe)
    ax.plot(closure_grid, pred, color=color, linewidth=2.5, label=label)

ax.axhline(260, color='red', linestyle='--', linewidth=2, label='MAOP (260 psig)')
ax.fill_between(closure_grid, 0, 260, alpha=0.1, color='green')

ax.set_xlabel('Valve Closure Time (seconds)', fontsize=12)
ax.set_ylabel('Peak Overpressure (psig)', fontsize=12)
ax.set_title('Surge vs. Closure Time', fontsize=14, pad=15)
ax.legend(loc='upper right', frameon=False, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('surge_curves.png', dpi=300, bbox_inches='tight')
print('✓ Visualization saved')

# ======================================================================
# Code Block 12
# ======================================================================

ΔP = ρ × a × Δv

# ======================================================================
# Code Block 13
# ======================================================================

"""
Generate synthetic transient scenarios with physics-inspired relationships.
"""
rng = np.random.default_rng(seed)

# ======================================================================
# Code Block 14
# ======================================================================

linepack = rng.uniform(0.6, 1.4, n_scenarios)         # Relative to normal (dimensionless)
closure_time = rng.uniform(0.2, 10.0, n_scenarios)    # Seconds
pump_trip = rng.integers(0, 2, n_scenarios)           # 0=no, 1=yes
velocity = rng.uniform(0.5, 3.0, n_scenarios)         # m/s
elevation_drop = rng.uniform(-80, 120, n_scenarios)   # m (negative = uphill)
temperature = rng.uniform(0, 35, n_scenarios)         # °C

# ======================================================================
# Code Block 15
# ======================================================================

base_surge = 35 * velocity / (1 + closure_time / 2.0)

# ======================================================================
# Code Block 16
# ======================================================================

static_head = 0.433 * (elevation_drop / 10.0)  # psi per 10m

# ======================================================================
# Code Block 17
# ======================================================================

pump_effect = pump_trip * (12 + 6 * np.tanh(3 * (1.5 - velocity)))

# ======================================================================
# Code Block 18
# ======================================================================

linepack_effect = 8 * (linepack - 1.0)

# ======================================================================
# Code Block 19
# ======================================================================

temp_effect = 0.2 * temperature

# ======================================================================
# Code Block 20
# ======================================================================

noise = rng.normal(0, 2.0, n_scenarios)

# ======================================================================
# Code Block 21
# ======================================================================

peak_overpress = (
    200 +  # Baseline operating pressure (psig)
    base_surge +
    static_head +
    pump_effect +
    linepack_effect +
    temp_effect +
    noise
)

df = pd.DataFrame({
    'linepack': linepack,
    'closure_time_s': closure_time,
    'pump_trip': pump_trip,
    'velocity_ms': velocity,
    'elevation_drop_m': elevation_drop,
    'temperature_c': temperature,
    'peak_overpress_psig': peak_overpress
})

return df

# ======================================================================
# Code Block 22
# ======================================================================

(df_train['closure_time_s'] < 0.5) &
(df_train['velocity_ms'] > 2.4) &
(df_train['pump_trip'] == 0) &
(df_train['linepack'].between(0.95, 1.05))

# ======================================================================
# Code Block 23
# ======================================================================

X, y, test_size=0.25, random_state=42

# ======================================================================
# Code Block 24
# ======================================================================

('scaler', StandardScaler()),
('gbr', HistGradientBoostingRegressor(
    max_depth=5,
    max_iter=500,
    learning_rate=0.05,
    random_state=42
))

# ======================================================================
# Code Block 25
# ======================================================================

velocity_ms,
linepack=1.0,
pump_trip=0,
elevation_drop_m=0,
temperature_c=15,
max_allowable_pressure=260

# ======================================================================
# Code Block 26
# ======================================================================

"""
Find minimum closure time that keeps peak pressure below MAOP.
"""
closure_times = np.linspace(0.2, 15, 150)

scenarios = pd.DataFrame({
    'linepack': linepack,
    'closure_time_s': closure_times,
    'pump_trip': pump_trip,
    'velocity_ms': velocity_ms,
    'elevation_drop_m': elevation_drop_m,
    'temperature_c': temperature_c
})

predicted_peaks = model.predict(scenarios)

# ======================================================================
# Code Block 27
# ======================================================================

safe_indices = np.where(predicted_peaks < max_allowable_pressure)[0]

if len(safe_indices) == 0:
    return None, predicted_peaks  # No safe closure time exists

min_safe_time = closure_times[safe_indices[0]]
return min_safe_time, predicted_peaks

# ======================================================================
# Code Block 28
# ======================================================================

velocity_ms=velocity,
linepack=1.1,  # Slightly overpacked
pump_trip=0,
max_allowable_pressure=260

# ======================================================================
# Code Block 29
# ======================================================================

probe = pd.DataFrame({
    'linepack': 1.0,
    'closure_time_s': closure_grid,
    'pump_trip': 0,
    'velocity_ms': v,
    'elevation_drop_m': 0,
    'temperature_c': 15
})

predicted = model.predict(probe)
ax.plot(closure_grid, predicted, color=color, linewidth=2.5, label=label)

# ======================================================================
# Code Block 30
# ======================================================================

dpdt = np.diff(pressure_trace[-window:]).mean()
dqdt = np.diff(flow_trace[-window:]).mean()

if dpdt < -5 and dqdt < -0.2:  # Thresholds from historical data
    return True
return False

# ======================================================================
# Code Block 31
# ======================================================================

safe_closure_time = predict_safe_closure_time(
    velocity_ms=current_velocity,
    pump_trip=1,  # Flag trip in model
    max_allowable_pressure=260
)
print(f'PUMP TRIP DETECTED: Use {safe_closure_time:.1f}s closure (extended)')

# ======================================================================
# Code Block 32
# ======================================================================

def __init__(self, num_features, hidden_dim):
    super().__init__()
    self.conv1 = GCNConv(num_features, hidden_dim)
    self.conv2 = GCNConv(hidden_dim, 1)  # Output: pressure at each node

def forward(self, x, edge_index):
    x = self.conv1(x, edge_index).relu()
    x = self.conv2(x, edge_index)
    return x

# ======================================================================
# Code Block 33
# ======================================================================

sample_idx = np.random.choice(len(X_train), len(X_train), replace=True)
X_boot = X_train.iloc[sample_idx]
y_boot = y_train.iloc[sample_idx]

# ======================================================================
# Code Block 34
# ======================================================================

model_i = GradientBoostingRegressor(random_state=i)
model_i.fit(X_boot, y_boot)

# ======================================================================
# Code Block 35
# ======================================================================

pred_i = model_i.predict(X_test)
predictions.append(pred_i)

# ======================================================================
# Code Block 36
# ======================================================================

"""
Log real event for model retraining.
"""
event_features = {
    'linepack': scada_data['linepack'],
    'closure_time_s': scada_data['closure_time'],
    'pump_trip': scada_data['pump_trip'],
    'velocity_ms': scada_data['velocity'],
    'elevation_drop_m': scada_data['elevation'],
    'temperature_c': scada_data['temperature'],
    'peak_overpress_psig': peak_pressure_measured
}

# ======================================================================
# Code Block 37
# ======================================================================

if len(historical_events) >= 20:
    retrain_model(historical_events)

# ======================================================================
# Code Block 38
# ======================================================================

rng = np.random.default_rng(seed)

linepack = rng.uniform(0.6, 1.4, n)
closure_time = rng.uniform(0.2, 10.0, n)
pump_trip = rng.integers(0, 2, n)
velocity = rng.uniform(0.5, 3.0, n)
elev = rng.uniform(-80, 120, n)
temp = rng.uniform(0, 35, n)

base = 35 * velocity / (1 + closure_time / 2.0)
head = 0.433 * (elev / 10.0)
trip = pump_trip * (12 + 6 * np.tanh(3 * (1.5 - velocity)))
pack = 8 * (linepack - 1.0)
noise = rng.normal(0, 2.0, n)

peak = 200 + base + head + trip + pack + 0.2 * temp + noise

return pd.DataFrame({
    'linepack': linepack,
    'closure_time_s': closure_time,
    'pump_trip': pump_trip,
    'velocity_ms': velocity,
    'elevation_drop_m': elev,
    'temperature_c': temp,
    'peak_overpress_psig': peak
})

# ======================================================================
# Code Block 39
# ======================================================================

('scaler', StandardScaler()),
('gbr', HistGradientBoostingRegressor(max_depth=5, max_iter=500, learning_rate=0.05, random_state=42))

# ======================================================================
# Code Block 40
# ======================================================================

times = np.linspace(0.2, 15, 150)
scenarios = pd.DataFrame({
    'linepack': linepack, 'closure_time_s': times, 'pump_trip': pump_trip,
    'velocity_ms': velocity, 'elevation_drop_m': elev, 'temperature_c': temp
})
peaks = model.predict(scenarios)
safe = np.where(peaks < maop)[0]
return times[safe[0]] if len(safe) > 0 else None, peaks

# ======================================================================
# Code Block 41
# ======================================================================

(1.5, '#f39c12', 'Medium (1.5 m/s)'),
                     (2.5, '#e74c3c', 'High (2.5 m/s)')]:
probe = pd.DataFrame({
    'linepack': 1.0, 'closure_time_s': closure_grid, 'pump_trip': 0,
    'velocity_ms': v, 'elevation_drop_m': 0, 'temperature_c': 15
})
pred = model.predict(probe)
ax.plot(closure_grid, pred, color=color, linewidth=2.5, label=label)
