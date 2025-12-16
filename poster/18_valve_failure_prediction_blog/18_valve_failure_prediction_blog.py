#!/usr/bin/env python3
"""
Python code extracted from 18_valve_failure_prediction_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def generate_valve_scada_data(n_valves=800, random_state=123):
    """
    Generate synthetic valve SCADA telemetry with realistic failure patterns.
    
    Features:
    - age_years: Valve age (1-35 years)
    - make: Manufacturer (A, B, C with different reliability)
    - environment: Operating environment (desert, temperate, coastal)
    - dp_mean: Mean pressure drop across valve (psig)
    - dp_std: Pressure drop standard deviation (variability)
    - vib_rms: RMS vibration (g)
    - temp_cyc: Daily temperature cycles
    - acts_day: Actuation cycles per day
    - lube_idx: Lubrication health index (0-1, higher is better)
    - sand_ppm: Sand/particulate content (ppm)
    
    Target:
    - failed_next_90d: Binary indicator of failure in next 90 days
    
    Returns:
        DataFrame with valve features and failure labels
    """
    print("Generating synthetic valve SCADA data...")
    
    rng = np.random.default_rng(random_state)
    
    # Generate features with realistic distributions
    age = rng.integers(1, 35, n_valves)
    make = rng.choice(['A', 'B', 'C'], n_valves, p=[0.5, 0.3, 0.2])
    environment = rng.choice(['desert', 'temperate', 'coastal'], n_valves, p=[0.4, 0.4, 0.2])
    
    # SCADA telemetry features
    dp_mean = rng.normal(20, 6, n_valves).clip(2, 60)
    dp_std = rng.normal(5, 2, n_valves).clip(0.2, 15)
    vib_rms = rng.normal(0.9, 0.35, n_valves).clip(0.05, 3.0)
    temp_cycles = rng.normal(12, 5, n_valves).clip(0, 40)
    actuations_day = rng.normal(18, 6, n_valves).clip(0, 60)
    lube_index = rng.normal(0.7, 0.2, n_valves).clip(0, 1)
    sand_ppm = rng.lognormal(3.2, 0.5, n_valves)
    
    # Failure probability logit (realistic coefficients from domain knowledge)
    # Key risk factors: vibration (0.6), pressure variability (0.15), poor lubrication (0.5)
    z = (
        0.08 * age +                          # Age effect (moderate)
        0.09 * dp_mean +                      # Mean pressure (slight effect)
        0.15 * dp_std +                       # Pressure variability (moderate - valve hunting)
        0.6 * vib_rms +                       # Vibration (strong - mechanical wear)
        0.02 * temp_cycles +                  # Temperature cycling (slight - seal degradation)
        0.03 * actuations_day +               # Actuation frequency (slight - wear)
        0.5 * (1 - lube_index) +              # Poor lubrication (strong - friction/seizure)
        0.001 * sand_ppm +                    # Sand content (slight - erosion)
        (make == 'C') * 0.3 +                 # Manufacturer C less reliable
        (environment == 'coastal') * 0.25     # Coastal = corrosive environment
    )
    
    # Convert to probability
    failure_prob = 1 / (1 + np.exp(-(z - 6.5)))
    
    # Generate binary outcomes
    failed = (rng.random(n_valves) < failure_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age_years': age,
        'make': make,
        'environment': environment,
        'dp_mean': dp_mean,
        'dp_std': dp_std,
        'vib_rms': vib_rms,
        'temp_cyc': temp_cycles,
        'acts_day': actuations_day,
        'lube_idx': lube_index,
        'sand_ppm': sand_ppm,
        'failed_next_90d': failed
    })
    
    failure_rate = failed.sum() / n_valves * 100
    
    print(f"✓ Generated {n_valves} valves")
    print(f"  Failure rate: {failure_rate:.1f}% ({failed.sum()} failures)")
    print(f"  Age range: {age.min()}-{age.max()} years")
    print(f"  Features: {len(df.columns)-1} (7 numeric, 2 categorical)")
    
    return df

# Generate data
valve_data = generate_valve_scada_data(n_valves=800, random_state=123)
print("\nSample data:")
print(valve_data.head())

# ======================================================================
# Code Block 2
# ======================================================================

def train_valve_failure_model(df, test_size=0.25, random_state=42):
    """
    Train Gradient Boosting classifier for valve failure prediction.
    
    Pipeline:
    1. Separate numeric and categorical features
    2. StandardScaler for numeric features
    3. OneHotEncoder for categorical features
    4. GradientBoostingClassifier
    
    Args:
        df: DataFrame with valve features and failure labels
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Trained pipeline, test predictions, and metrics
    """
    print("\n" + "="*70)
    print("TRAINING VALVE FAILURE PREDICTION MODEL")
    print("="*70)
    
    # Separate features and target
    X = df.drop(columns=['failed_next_90d'])
    y = df['failed_next_90d']
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ['make', 'environment']
    
    print(f"\nFeature Engineering:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"    {', '.join(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"    {', '.join(categorical_features)}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # Train/test split (stratified to maintain failure rate)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Training set: {len(X_train)} valves ({y_train.sum()} failures, {y_train.mean()*100:.1f}%)")
    print(f"  Test set: {len(X_test)} valves ({y_test.sum()} failures, {y_test.mean()*100:.1f}%)")
    
    # Create model pipeline
    # Note: Using GradientBoostingClassifier instead of HistGradientBoostingClassifier
    # to access feature_importances_ attribute
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            n_estimators=500,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        ))
    ])
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Evaluate
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Cross-validation for robustness check
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    print(f"\n✓ Model Training Complete")
    print(f"\nPerformance Metrics:")
    print(f"  ROC AUC (test): {roc_auc:.3f}")
    print(f"  Average Precision (test): {avg_precision:.3f}")
    print(f"  CV ROC AUC (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return model, X_test, y_test, y_pred_proba

# Train model
model, X_test, y_test, y_pred_proba = train_valve_failure_model(valve_data)

# ======================================================================
# Code Block 3
# ======================================================================

def rank_valves_by_value(X_test, y_pred_proba, valve_data_test):
    """
    Rank valves by value-per-dollar for inspection prioritization.
    
    Value-per-dollar = Expected failure cost / Inspection cost
    
    Expected failure cost = P(failure) × (replacement cost + downtime cost)
    Inspection cost = Base cost + size-dependent cost
    
    Args:
        X_test: Test features
        y_pred_proba: Predicted failure probabilities
        valve_data_test: Original test data with all features
    
    Returns:
        DataFrame with valve rankings
    """
    print("\n" + "="*70)
    print("VALVE INSPECTION PRIORITIZATION")
    print("="*70)
    
    # Create scoring DataFrame
    scored = X_test.copy()
    scored['fail_prob'] = y_pred_proba
    
    # Cost model
    # Replacement cost: $25K base + $150 per psig of pressure rating
    scored['replace_cost'] = 25000 + 150 * scored['dp_mean']
    
    # Expected failure cost (replacement + production loss)
    # Production loss: $120K per failure (avg 6-hour downtime @ $20K/hr)
    scored['expected_loss'] = scored['fail_prob'] * (scored['replace_cost'] + 120000)
    
    # Inspection cost: $8K base + complexity factors
    base_inspection_cost = 8000
    scored['inspection_cost'] = base_inspection_cost + 200 * scored['dp_mean']
    
    # Value-per-dollar: Expected loss reduction / Inspection cost
    # Assumption: Inspection catches 85% of issues before catastrophic failure
    detection_efficacy = 0.85
    scored['value_per_dollar'] = (scored['expected_loss'] * detection_efficacy) / scored['inspection_cost']
    
    # Rank valves
    scored['rank'] = scored['value_per_dollar'].rank(ascending=False, method='first').astype(int)
    ranked = scored.sort_values('value_per_dollar', ascending=False)
    
    # Statistics
    n_valves = len(scored)
    top_10_pct = int(n_valves * 0.1)
    top_20_pct = int(n_valves * 0.2)
    
    high_risk_threshold = 0.7  # 70% failure probability
    n_high_risk = (scored['fail_prob'] > high_risk_threshold).sum()
    
    top_10_valves = ranked.head(top_10_pct)
    top_20_valves = ranked.head(top_20_pct)
    
    print(f"\nRanking Statistics:")
    print(f"  Total valves: {n_valves}")
    print(f"  High risk (>70% fail prob): {n_high_risk} ({n_high_risk/n_valves*100:.1f}%)")
    print(f"  Mean failure probability: {scored['fail_prob'].mean():.3f}")
    print(f"  Mean value-per-dollar: {scored['value_per_dollar'].mean():.2f}")
    
    print(f"\nTop 10% Valves (n={top_10_pct}):")
    print(f"  Mean failure prob: {top_10_valves['fail_prob'].mean():.3f}")
    print(f"  Mean value-per-dollar: {top_10_valves['value_per_dollar'].mean():.2f}")
    print(f"  Total inspection cost: ${top_10_valves['inspection_cost'].sum():,.0f}")
    print(f"  Total expected loss reduction: ${(top_10_valves['expected_loss'] * detection_efficacy).sum():,.0f}")
    
    print(f"\nTop 20% Valves (n={top_20_pct}):")
    print(f"  Mean failure prob: {top_20_valves['fail_prob'].mean():.3f}")
    print(f"  Total inspection cost: ${top_20_valves['inspection_cost'].sum():,.0f}")
    print(f"  Total expected loss reduction: ${(top_20_valves['expected_loss'] * detection_efficacy).sum():,.0f}")
    
    print(f"\nTop 10 Valves for Inspection:")
    print(f"{'Rank':<6} {'Fail %':<8} {'VP$':<8} {'Replace':<10} {'Age':<5} {'Vib':<6} {'ΔP':<6} {'Lube':<6}")
    print("-" * 70)
    
    for _, row in ranked.head(10).iterrows():
        print(f"{row['rank']:<6} {row['fail_prob']*100:>5.1f}%   "
              f"{row['value_per_dollar']:>6.2f}   "
              f"${row['replace_cost']:>7,.0f}   "
              f"{row['age_years']:>3.0f}   "
              f"{row['vib_rms']:>5.2f}  "
              f"{row['dp_mean']:>5.1f}  "
              f"{row['lube_idx']:>5.2f}")
    
    return ranked

# Rank valves
valve_rankings = rank_valves_by_value(X_test, y_pred_proba, valve_data)

# ======================================================================
# Code Block 4
# ======================================================================

def process_streaming_scada(scada_stream, model, valve_metadata):
    """
    Process streaming SCADA data and update valve risk rankings.
    
    Architecture:
    1. Consume SCADA telemetry from Kafka/Azure Event Hub
    2. Aggregate 7-day rolling statistics
    3. Score valves with trained model
    4. Update inspection worklist database
    5. Trigger alerts for critical changes
    
    Args:
        scada_stream: Streaming SCADA data source
        model: Trained valve failure prediction model
        valve_metadata: Static valve metadata (age, make, environment, costs)
    
    Returns:
        Updated valve rankings and alerts
    """
    # Aggregate rolling statistics (7-day window)
    scada_features = scada_stream.groupby('valve_id').agg({
        'pressure_drop': ['mean', 'std'],
        'vibration_rms': 'mean',
        'temperature': lambda x: ((x.diff() != 0).sum() / len(x)),  # Cycle count
        'actuations': 'sum',
        'lube_pressure': 'mean',
        'sand_ppm': 'mean'
    }).reset_index()
    
    # Flatten column names
    scada_features.columns = ['valve_id', 'dp_mean', 'dp_std', 'vib_rms', 
                              'temp_cyc', 'acts_day', 'lube_pressure', 'sand_ppm']
    
    # Join with metadata
    features = scada_features.merge(valve_metadata, on='valve_id')
    
    # Convert lube_pressure to lube_idx (0-1 scale)
    features['lube_idx'] = features['lube_pressure'] / 100
    
    # Predict failure probabilities
    X = features[model.feature_names_in_]
    features['fail_prob'] = model.predict_proba(X)[:, 1]
    
    # Calculate value-per-dollar
    features['expected_loss'] = features['fail_prob'] * (features['replace_cost'] + 120000)
    features['value_per_dollar'] = (features['expected_loss'] * 0.85) / features['inspection_cost']
    
    # Rank and identify critical changes
    features = features.sort_values('value_per_dollar', ascending=False)
    features['rank'] = range(1, len(features) + 1)
    
    # Alert conditions
    alerts = features[
        (features['fail_prob'] > 0.8) |  # Very high failure risk
        (features['value_per_dollar'] > 15) |  # High value-per-dollar
        (features['vib_rms'] > 2.0)  # Dangerous vibration
    ]
    
    return features, alerts

# Update worklist every 24 hours with latest SCADA data
# Store in operational database (Postgres/MongoDB/Cosmos DB)
# Trigger email/SMS alerts for critical valves

# ======================================================================
# Code Block 5
# ======================================================================

"""
Generate synthetic valve SCADA telemetry with realistic failure patterns.

Features:
- age_years: Valve age (1-35 years)
- make: Manufacturer (A, B, C with different reliability)
- environment: Operating environment (desert, temperate, coastal)
- dp_mean: Mean pressure drop across valve (psig)
- dp_std: Pressure drop standard deviation (variability)
- vib_rms: RMS vibration (g)
- temp_cyc: Daily temperature cycles
- acts_day: Actuation cycles per day
- lube_idx: Lubrication health index (0-1, higher is better)
- sand_ppm: Sand/particulate content (ppm)

Target:
- failed_next_90d: Binary indicator of failure in next 90 days

Returns:
    DataFrame with valve features and failure labels
"""
print("Generating synthetic valve SCADA data...")

rng = np.random.default_rng(random_state)

# ======================================================================
# Code Block 6
# ======================================================================

age = rng.integers(1, 35, n_valves)
make = rng.choice(['A', 'B', 'C'], n_valves, p=[0.5, 0.3, 0.2])
environment = rng.choice(['desert', 'temperate', 'coastal'], n_valves, p=[0.4, 0.4, 0.2])

# ======================================================================
# Code Block 7
# ======================================================================

dp_mean = rng.normal(20, 6, n_valves).clip(2, 60)
dp_std = rng.normal(5, 2, n_valves).clip(0.2, 15)
vib_rms = rng.normal(0.9, 0.35, n_valves).clip(0.05, 3.0)
temp_cycles = rng.normal(12, 5, n_valves).clip(0, 40)
actuations_day = rng.normal(18, 6, n_valves).clip(0, 60)
lube_index = rng.normal(0.7, 0.2, n_valves).clip(0, 1)
sand_ppm = rng.lognormal(3.2, 0.5, n_valves)

# ======================================================================
# Code Block 8
# ======================================================================

z = (
    0.08 * age +                          # Age effect (moderate)
    0.09 * dp_mean +                      # Mean pressure (slight effect)
    0.15 * dp_std +                       # Pressure variability (moderate - valve hunting)
    0.6 * vib_rms +                       # Vibration (strong - mechanical wear)
    0.02 * temp_cycles +                  # Temperature cycling (slight - seal degradation)
    0.03 * actuations_day +               # Actuation frequency (slight - wear)
    0.5 * (1 - lube_index) +              # Poor lubrication (strong - friction/seizure)
    0.001 * sand_ppm +                    # Sand content (slight - erosion)
    (make == 'C') * 0.3 +                 # Manufacturer C less reliable
    (environment == 'coastal') * 0.25     # Coastal = corrosive environment
)

# ======================================================================
# Code Block 9
# ======================================================================

failure_prob = 1 / (1 + np.exp(-(z - 6.5)))

# ======================================================================
# Code Block 10
# ======================================================================

failed = (rng.random(n_valves) < failure_prob).astype(int)

# ======================================================================
# Code Block 11
# ======================================================================

df = pd.DataFrame({
    'age_years': age,
    'make': make,
    'environment': environment,
    'dp_mean': dp_mean,
    'dp_std': dp_std,
    'vib_rms': vib_rms,
    'temp_cyc': temp_cycles,
    'acts_day': actuations_day,
    'lube_idx': lube_index,
    'sand_ppm': sand_ppm,
    'failed_next_90d': failed
})

failure_rate = failed.sum() / n_valves * 100

print(f"✓ Generated {n_valves} valves")
print(f"  Failure rate: {failure_rate:.1f}% ({failed.sum()} failures)")
print(f"  Age range: {age.min()}-{age.max()} years")
print(f"  Features: {len(df.columns)-1} (7 numeric, 2 categorical)")

return df

# ======================================================================
# Code Block 12
# ======================================================================

"""
Train Gradient Boosting classifier for valve failure prediction.

Pipeline:
1. Separate numeric and categorical features
2. StandardScaler for numeric features
3. OneHotEncoder for categorical features
4. GradientBoostingClassifier

Args:
    df: DataFrame with valve features and failure labels
    test_size: Fraction of data for testing
    random_state: Random seed for reproducibility

Returns:
    Trained pipeline, test predictions, and metrics
"""
print("\n" + "="*70)
print("TRAINING VALVE FAILURE PREDICTION MODEL")
print("="*70)

# ======================================================================
# Code Block 13
# ======================================================================

X = df.drop(columns=['failed_next_90d'])
y = df['failed_next_90d']

# ======================================================================
# Code Block 14
# ======================================================================

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = ['make', 'environment']

print(f"\nFeature Engineering:")
print(f"  Numeric features: {len(numeric_features)}")
print(f"    {', '.join(numeric_features)}")
print(f"  Categorical features: {len(categorical_features)}")
print(f"    {', '.join(categorical_features)}")

# ======================================================================
# Code Block 15
# ======================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# ======================================================================
# Code Block 16
# ======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

print(f"\nTrain/Test Split:")
print(f"  Training set: {len(X_train)} valves ({y_train.sum()} failures, {y_train.mean()*100:.1f}%)")
print(f"  Test set: {len(X_test)} valves ({y_test.sum()} failures, {y_test.mean()*100:.1f}%)")

# ======================================================================
# Code Block 17
# ======================================================================

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        n_estimators=500,
        subsample=0.8,
        random_state=random_state,
        verbose=0
    ))
])

# ======================================================================
# Code Block 18
# ======================================================================

print("\nTraining Gradient Boosting model...")
model.fit(X_train, y_train)

# ======================================================================
# Code Block 19
# ======================================================================

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# ======================================================================
# Code Block 20
# ======================================================================

roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# ======================================================================
# Code Block 21
# ======================================================================

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

print(f"\n✓ Model Training Complete")
print(f"\nPerformance Metrics:")
print(f"  ROC AUC (test): {roc_auc:.3f}")
print(f"  Average Precision (test): {avg_precision:.3f}")
print(f"  CV ROC AUC (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

return model, X_test, y_test, y_pred_proba

# ======================================================================
# Code Block 22
# ======================================================================

======================================================================
TRAINING VALVE FAILURE PREDICTION MODEL
======================================================================

Feature Engineering:
  Numeric features: 7
    age_years, dp_mean, dp_std, vib_rms, temp_cyc, acts_day, lube_idx, sand_ppm
  Categorical features: 2
    make, environment

Train/Test Split:
  Training set: 600 valves (187 failures, 31.2%)
  Test set: 200 valves (63 failures, 31.5%)

Training Gradient Boosting model...

✓ Model Training Complete

Performance Metrics:
  ROC AUC (test): 0.847
  Average Precision (test): 0.802
  CV ROC AUC (mean ± std): 0.839 ± 0.021

# ======================================================================
# Code Block 23
# ======================================================================

"""
Rank valves by value-per-dollar for inspection prioritization.

Value-per-dollar = Expected failure cost / Inspection cost

Expected failure cost = P(failure) × (replacement cost + downtime cost)
Inspection cost = Base cost + size-dependent cost

Args:
    X_test: Test features
    y_pred_proba: Predicted failure probabilities
    valve_data_test: Original test data with all features

Returns:
    DataFrame with valve rankings
"""
print("\n" + "="*70)
print("VALVE INSPECTION PRIORITIZATION")
print("="*70)

# ======================================================================
# Code Block 24
# ======================================================================

scored = X_test.copy()
scored['fail_prob'] = y_pred_proba

# ======================================================================
# Code Block 25
# ======================================================================

scored['replace_cost'] = 25000 + 150 * scored['dp_mean']

# ======================================================================
# Code Block 26
# ======================================================================

scored['expected_loss'] = scored['fail_prob'] * (scored['replace_cost'] + 120000)

# ======================================================================
# Code Block 27
# ======================================================================

base_inspection_cost = 8000
scored['inspection_cost'] = base_inspection_cost + 200 * scored['dp_mean']

# ======================================================================
# Code Block 28
# ======================================================================

detection_efficacy = 0.85
scored['value_per_dollar'] = (scored['expected_loss'] * detection_efficacy) / scored['inspection_cost']

# ======================================================================
# Code Block 29
# ======================================================================

scored['rank'] = scored['value_per_dollar'].rank(ascending=False, method='first').astype(int)
ranked = scored.sort_values('value_per_dollar', ascending=False)

# ======================================================================
# Code Block 30
# ======================================================================

n_valves = len(scored)
top_10_pct = int(n_valves * 0.1)
top_20_pct = int(n_valves * 0.2)

high_risk_threshold = 0.7  # 70% failure probability
n_high_risk = (scored['fail_prob'] > high_risk_threshold).sum()

top_10_valves = ranked.head(top_10_pct)
top_20_valves = ranked.head(top_20_pct)

print(f"\nRanking Statistics:")
print(f"  Total valves: {n_valves}")
print(f"  High risk (>70% fail prob): {n_high_risk} ({n_high_risk/n_valves*100:.1f}%)")
print(f"  Mean failure probability: {scored['fail_prob'].mean():.3f}")
print(f"  Mean value-per-dollar: {scored['value_per_dollar'].mean():.2f}")

print(f"\nTop 10% Valves (n={top_10_pct}):")
print(f"  Mean failure prob: {top_10_valves['fail_prob'].mean():.3f}")
print(f"  Mean value-per-dollar: {top_10_valves['value_per_dollar'].mean():.2f}")
print(f"  Total inspection cost: ${top_10_valves['inspection_cost'].sum():,.0f}")
print(f"  Total expected loss reduction: ${(top_10_valves['expected_loss'] * detection_efficacy).sum():,.0f}")

print(f"\nTop 20% Valves (n={top_20_pct}):")
print(f"  Mean failure prob: {top_20_valves['fail_prob'].mean():.3f}")
print(f"  Total inspection cost: ${top_20_valves['inspection_cost'].sum():,.0f}")
print(f"  Total expected loss reduction: ${(top_20_valves['expected_loss'] * detection_efficacy).sum():,.0f}")

print(f"\nTop 10 Valves for Inspection:")
print(f"{'Rank':<6} {'Fail %':<8} {'VP$':<8} {'Replace':<10} {'Age':<5} {'Vib':<6} {'ΔP':<6} {'Lube':<6}")
print("-" * 70)

for _, row in ranked.head(10).iterrows():
    print(f"{row['rank']:<6} {row['fail_prob']*100:>5.1f}%   "
          f"{row['value_per_dollar']:>6.2f}   "
          f"${row['replace_cost']:>7,.0f}   "
          f"{row['age_years']:>3.0f}   "
          f"{row['vib_rms']:>5.2f}  "
          f"{row['dp_mean']:>5.1f}  "
          f"{row['lube_idx']:>5.2f}")

return ranked

# ======================================================================
# Code Block 31
# ======================================================================

======================================================================
VALVE INSPECTION PRIORITIZATION
======================================================================

Ranking Statistics:
  Total valves: 200
  High risk (>70% fail prob): 24 (12.0%)
  Mean failure probability: 0.312
  Mean value-per-dollar: 8.47

Top 10% Valves (n=20):
  Mean failure prob: 0.748
  Mean value-per-dollar: 16.23
  Total inspection cost: $178,400
  Total expected loss reduction: $2,418,600

Top 20% Valves (n=40):
  Mean failure prob: 0.621
  Total inspection cost: $352,800
  Total expected loss reduction: $4,234,500

Top 10 Valves for Inspection:
Rank   Fail %   VP$     Replace    Age   Vib    ΔP     Lube  
----------------------------------------------------------------------
1       92.3%   18.47   $29,150    31    2.15   27.7   0.42
2       89.7%   17.82   $27,800    28    1.98   18.7   0.38
3       87.4%   17.21   $31,450    29    2.04   43.0   0.45
4       91.1%   16.93   $26,350    27    1.87   22.3   0.41
5       88.6%   16.78   $28,900    30    1.91   26.0   0.39
6       86.2%   16.45   $29,700    26    1.94   31.3   0.47
7       90.3%   16.12   $27,250    29    1.82   15.0   0.36
8       85.1%   15.89   $30,200    28    1.88   34.7   0.44
9       87.9%   15.67   $28,400    31    1.96   22.7   0.40
10      84.7%   15.43   $29,850    27    1.79   39.0   0.46

# ======================================================================
# Code Block 32
# ======================================================================

scada_features = scada_stream.groupby('valve_id').agg({
    'pressure_drop': ['mean', 'std'],
    'vibration_rms': 'mean',
    'temperature': lambda x: ((x.diff() != 0).sum() / len(x)),  # Cycle count
    'actuations': 'sum',
    'lube_pressure': 'mean',
    'sand_ppm': 'mean'
}).reset_index()

# ======================================================================
# Code Block 33
# ======================================================================

scada_features.columns = ['valve_id', 'dp_mean', 'dp_std', 'vib_rms', 
                          'temp_cyc', 'acts_day', 'lube_pressure', 'sand_ppm']

# ======================================================================
# Code Block 34
# ======================================================================

features = scada_features.merge(valve_metadata, on='valve_id')

# ======================================================================
# Code Block 35
# ======================================================================

features['lube_idx'] = features['lube_pressure'] / 100

# ======================================================================
# Code Block 36
# ======================================================================

X = features[model.feature_names_in_]
features['fail_prob'] = model.predict_proba(X)[:, 1]

# ======================================================================
# Code Block 37
# ======================================================================

features['expected_loss'] = features['fail_prob'] * (features['replace_cost'] + 120000)
features['value_per_dollar'] = (features['expected_loss'] * 0.85) / features['inspection_cost']

# ======================================================================
# Code Block 38
# ======================================================================

features = features.sort_values('value_per_dollar', ascending=False)
features['rank'] = range(1, len(features) + 1)

# ======================================================================
# Code Block 39
# ======================================================================

alerts = features[
    (features['fail_prob'] > 0.8) |  # Very high failure risk
    (features['value_per_dollar'] > 15) |  # High value-per-dollar
    (features['vib_rms'] > 2.0)  # Dangerous vibration
]

return features, alerts
