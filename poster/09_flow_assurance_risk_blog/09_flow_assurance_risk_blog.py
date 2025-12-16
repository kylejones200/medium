#!/usr/bin/env python3
"""
Python code extracted from 09_flow_assurance_risk_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def generate_pipeline_telemetry(n_segments=4000, seed=77):
    """
    Generate realistic pipeline segment telemetry data.
    
    Simulates subsea working system with multiple crude types,
    varying thermal conditions, and flow regimes.
    
    Parameters:
    -----------
    n_segments : int
        Number of pipeline segments to simulate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame : Pipeline telemetry with flow assurance parameters
    """
    rng = np.random.default_rng(seed)
    
    # Crude oil types with different wax appearance temperatures (WAT)
    crude_types = ['light_sweet', 'medium', 'heavy_waxy']
    crude_probs = [0.5, 0.35, 0.15]  # Production mix
    crude = rng.choice(crude_types, n_segments, p=crude_probs)
    
    # Wax Appearance Temperature (°C) - temperature below which wax precipitates
    # Heavy waxy crudes have higher WAT (~35°C), lighter crudes lower (~27°C)
    wat_celsius = np.where(
        crude == 'heavy_waxy',
        rng.normal(35, 2, n_segments),
        rng.normal(27, 2, n_segments)
    )
    
    # Pipeline inlet and outlet temperatures
    temp_in_celsius = rng.normal(32, 4, n_segments)  # Wellhead temperature
    temp_out_celsius = temp_in_celsius - rng.normal(3, 2, n_segments).clip(0.5, 10)  # Cooling along pipeline
    
    # Operating pressure (bar)
    pressure_bar = rng.normal(55, 6, n_segments)
    
    # Flow rate (thousands of standard cubic meters per hour)
    flow_ksm3h = rng.normal(2.0, 0.6, n_segments).clip(0.3, 4.0)
    
    # Shear rate proxy (flow velocity / viscosity factor)
    # Higher shear inhibits wax deposition
    shear_proxy = 0.3 * flow_ksm3h / (pressure_bar / 50)
    
    # Chemical inhibitor injection (0 = no injection, 1 = active)
    inhibitor_active = rng.choice([0, 1], n_segments, p=[0.7, 0.3])
    
    # Water cut (fraction of produced water in flow stream)
    water_cut = rng.beta(2, 10, n_segments)  # Typically low but some wells have high water
    
    # Calculate thermal margin (WAT - average pipeline temperature)
    # Positive margin = temperature below WAT = wax risk
    avg_temp = 0.5 * (temp_in_celsius + temp_out_celsius)
    thermal_margin = wat_celsius - avg_temp
    
    # Risk probability based on physical mechanisms
    # Higher risk when:
    # - Thermal margin is positive (temp below WAT)
    # - High water cut (hydrate risk)
    # - Heavy waxy crude
    # - No inhibitor
    # - Low shear (allows deposition)
    base_logit = (
        0.8 * (thermal_margin > 0) +      # Below WAT
        0.4 * (thermal_margin > 2) +      # Well below WAT
        0.3 * (water_cut > 0.2) +         # High water content
        0.3 * (crude == 'heavy_waxy') -   # Waxy crude
        0.4 * inhibitor_active -           # Chemical protection
        0.3 * (shear_proxy > 0.9)         # High shear prevents deposition
    )
    
    # Convert to probability (logistic function)
    risk_probability = 1 / (1 + np.exp(-(base_logit - 0.4)))
    
    # Generate binary risk labels
    risk_observed = (rng.random(n_segments) < risk_probability).astype(int)
    
    # Create DataFrame
    telemetry = pd.DataFrame({
        'crude_type': crude,
        'wat_celsius': wat_celsius,
        'temp_in_celsius': temp_in_celsius,
        'temp_out_celsius': temp_out_celsius,
        'pressure_bar': pressure_bar,
        'flow_ksm3h': flow_ksm3h,
        'shear_proxy': shear_proxy,
        'inhibitor_active': inhibitor_active,
        'water_cut': water_cut,
        'risk_observed': risk_observed
    })
    
    return telemetry

# Generate synthetic telemetry
pipeline_data = generate_pipeline_telemetry(n_segments=4000)

print(f"Generated {len(pipeline_data)} pipeline segments")
print(f"\nCrude Type Distribution:")
print(pipeline_data['crude_type'].value_counts())
print(f"\nRisk Rate: {pipeline_data['risk_observed'].mean():.1%}")
print(f"\nKey Statistics:")
print(f"  WAT Range: {pipeline_data['wat_celsius'].min():.1f} to {pipeline_data['wat_celsius'].max():.1f}°C")
print(f"  Temp Range: {pipeline_data['temp_out_celsius'].min():.1f} to {pipeline_data['temp_in_celsius'].max():.1f}°C")
print(f"  Pressure Range: {pipeline_data['pressure_bar'].min():.1f} to {pipeline_data['pressure_bar'].max():.1f} bar")

# Display sample data
print("\nSample Pipeline Segments:")
print(pipeline_data.head())

# ======================================================================
# Code Block 2
# ======================================================================

def train_risk_classifier(telemetry_data, test_size=0.25, random_state=42):
    """
    Train Random Forest classifier for flow assurance risk prediction.
    
    Uses ensemble methods to capture nonlinear interactions between
    thermal, hydraulic, and compositional variables.
    
    Parameters:
    -----------
    telemetry_data : pd.DataFrame
        Pipeline telemetry with risk labels
    test_size : float
        Fraction of data for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Trained model, predictions, and performance metrics
    """
    # Separate features and target
    X = telemetry_data.drop(columns=['risk_observed'])
    y = telemetry_data['risk_observed']
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ['crude_type']
    
    # Build preprocessing pipeline (Pythonic with ColumnTransformer)
    from sklearn.preprocessing import OneHotEncoder
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', StandardScaler(), numeric_features),
            ('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Build full pipeline with Random Forest
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=3,
            n_jobs=-1
        ))
    ])
    
    # Train model
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates (Pythonic safe division)
    sensitivity = tp / max(1, tp + fn)  # True positive rate
    specificity = tn / max(1, tn + fp)  # True negative rate
    precision = tp / max(1, tp + fp)    # Positive predictive value
    npv = tn / max(1, tn + fn)          # Negative predictive value
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

# Train model
results = train_risk_classifier(pipeline_data)

print("\nFlow Assurance Risk Model Performance:")
print("=" * 60)
print(f"ROC AUC Score: {results['roc_auc']:.3f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {results['confusion_matrix'][0,0]:,}")
print(f"  False Positives: {results['confusion_matrix'][0,1]:,}")
print(f"  False Negatives: {results['confusion_matrix'][1,0]:,}")
print(f"  True Positives:  {results['confusion_matrix'][1,1]:,}")
print(f"\nDiagnostic Metrics:")
print(f"  Sensitivity (Recall):    {results['sensitivity']:.1%}")
print(f"  Specificity:             {results['specificity']:.1%}")
print(f"  Precision:               {results['precision']:.1%}")
print(f"  Negative Predictive Val: {results['npv']:.1%}")
print(f"\nInterpretation:")
print(f"  - Model correctly identifies {results['sensitivity']:.0%} of actual risk events")
print(f"  - Only {(1-results['specificity']):.0%} false alarm rate on safe segments")

# ======================================================================
# Code Block 3
# ======================================================================

def prioritize_high_risk_segments(model_results, telemetry_data, top_n=20):
    """
    Rank pipeline segments by risk-adjusted criticality.
    
    Combines predicted risk probability with operational criticality
    (pressure × flow rate) to prioritize inspection and intervention.
    
    Parameters:
    -----------
    model_results : dict
        Trained model and predictions
    telemetry_data : pd.DataFrame
        Full pipeline telemetry
    top_n : int
        Number of top segments to return
    
    Returns:
    --------
    pd.DataFrame : Ranked high-risk segments with recommendations
    """
    # Get test set with predictions
    test_segments = model_results['X_test'].copy()
    test_segments['risk_probability'] = model_results['y_proba']
    test_segments['actual_risk'] = model_results['y_test'].values
    
    # Calculate operational criticality (Pythonic weighted sum)
    # Higher pressure and flow = more critical segment
    test_segments['criticality_score'] = (
        test_segments['pressure_bar'] * 0.6 +
        test_segments['flow_ksm3h'] * 10.0  # Scale flow to comparable range
    )
    
    # Combined risk ranking (Pythonic)
    # Risk probability × (1 + criticality factor)
    test_segments['risk_rank_score'] = (
        test_segments['risk_probability'] * 
        (1 + 0.003 * test_segments['criticality_score'])
    )
    
    # Calculate thermal margin for reporting
    test_segments['thermal_margin'] = (
        test_segments['wat_celsius'] - 
        0.5 * (test_segments['temp_in_celsius'] + test_segments['temp_out_celsius'])
    )
    
    # Risk classification (Pythonic with pd.cut)
    test_segments['risk_category'] = pd.cut(
        test_segments['risk_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    # Recommended action mapping (Pythonic dictionary)
    action_map = {
        'LOW': 'Continue monitoring',
        'MEDIUM': 'Increase inspection frequency',
        'HIGH': 'Priority intervention - consider pigging or inhibitor boost'
    }
    test_segments['recommended_action'] = test_segments['risk_category'].map(action_map)
    
    # Sort by risk rank and select top segments
    top_segments = test_segments.sort_values('risk_rank_score', ascending=False).head(top_n)
    
    # Select key columns for reporting
    report_columns = [
        'risk_probability', 'risk_rank_score', 'risk_category',
        'crude_type', 'wat_celsius', 'thermal_margin',
        'temp_in_celsius', 'temp_out_celsius',
        'pressure_bar', 'flow_ksm3h',
        'inhibitor_active', 'water_cut',
        'recommended_action'
    ]
    
    return top_segments[report_columns]

# Generate prioritized worklist
priority_segments = prioritize_high_risk_segments(results, pipeline_data, top_n=20)

print("\nTop 10 Highest-Risk Pipeline Segments:")
print("=" * 60)
for idx, (i, row) in enumerate(priority_segments.head(10).iterrows(), 1):
    print(f"\nSegment {idx} (Risk Score: {row['risk_rank_score']:.3f}):")
    print(f"  Risk Probability: {row['risk_probability']:.1%} ({row['risk_category']})")
    print(f"  Crude Type: {row['crude_type']}")
    print(f"  WAT: {row['wat_celsius']:.1f}°C, Thermal Margin: {row['thermal_margin']:+.1f}°C")
    print(f"  Temp In: {row['temp_in_celsius']:.1f}°C, Temp Out: {row['temp_out_celsius']:.1f}°C")
    print(f"  Pressure: {row['pressure_bar']:.1f} bar, Flow: {row['flow_ksm3h']:.2f} kSm³/h")
    print(f"  Inhibitor: {'Active' if row['inhibitor_active'] else 'Inactive'}")
    print(f"  Water Cut: {row['water_cut']:.1%}")
    print(f"  Recommendation: {row['recommended_action']}")

# Summary statistics by risk category
print("\n\nRisk Category Distribution:")
print("=" * 60)
risk_summary = priority_segments.groupby('risk_category').agg({
    'risk_probability': ['count', 'mean'],
    'inhibitor_active': 'mean',
    'thermal_margin': 'mean'
}).round(3)
print(risk_summary)

# ======================================================================
# Code Block 4
# ======================================================================

def analyze_feature_importance(model_results, feature_names):
    """
    Extract and analyze feature importance from Random Forest model.
    
    Identifies which operational parameters have greatest influence
    on flow assurance risk prediction.
    
    Parameters:
    -----------
    model_results : dict
        Trained model results
    feature_names : list
        Names of input features
    
    Returns:
    --------
    pd.DataFrame : Ranked feature importance scores
    """
    # Extract Random Forest from pipeline
    rf_model = model_results['model'].named_steps['classifier']
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create importance DataFrame (Pythonic)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    
    return importance_df

# Get feature names (after preprocessing)
# Note: OneHotEncoder creates binary features for each category (minus reference)
numeric_cols = results['X_test'].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = ['crude_type_light_sweet', 'crude_type_medium']  # heavy_waxy is reference
feature_names = numeric_cols + categorical_cols

# Analyze importance
feature_importance = analyze_feature_importance(results, feature_names)

print("\nFeature Importance Analysis:")
print("=" * 60)
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:20s}: {row['importance']:.3f} ({row['cumulative_importance']:.1%} cumulative)")

print("\nKey Insights:")
top_features = feature_importance.head(3)['feature'].tolist()
print(f"  - Top 3 features ({', '.join(top_features)}) explain")
print(f"    {feature_importance.head(3)['cumulative_importance'].iloc[-1]:.0%} of risk variation")
print(f"  - Focus monitoring and control on these parameters for maximum impact")

# ======================================================================
# Code Block 5
# ======================================================================

"""
Generate realistic pipeline segment telemetry data.

Simulates subsea working system with multiple crude types,
varying thermal conditions, and flow regimes.

Parameters:
-----------
n_segments : int
    Number of pipeline segments to simulate
seed : int
    Random seed for reproducibility

Returns:
--------
pd.DataFrame : Pipeline telemetry with flow assurance parameters
"""
rng = np.random.default_rng(seed)

# ======================================================================
# Code Block 6
# ======================================================================

crude_types = ['light_sweet', 'medium', 'heavy_waxy']
crude_probs = [0.5, 0.35, 0.15]  # Production mix
crude = rng.choice(crude_types, n_segments, p=crude_probs)

# ======================================================================
# Code Block 7
# ======================================================================

wat_celsius = np.where(
    crude == 'heavy_waxy',
    rng.normal(35, 2, n_segments),
    rng.normal(27, 2, n_segments)
)

# ======================================================================
# Code Block 8
# ======================================================================

temp_in_celsius = rng.normal(32, 4, n_segments)  # Wellhead temperature
temp_out_celsius = temp_in_celsius - rng.normal(3, 2, n_segments).clip(0.5, 10)  # Cooling along pipeline

# ======================================================================
# Code Block 9
# ======================================================================

pressure_bar = rng.normal(55, 6, n_segments)

# ======================================================================
# Code Block 10
# ======================================================================

flow_ksm3h = rng.normal(2.0, 0.6, n_segments).clip(0.3, 4.0)

# ======================================================================
# Code Block 11
# ======================================================================

shear_proxy = 0.3 * flow_ksm3h / (pressure_bar / 50)

# ======================================================================
# Code Block 12
# ======================================================================

inhibitor_active = rng.choice([0, 1], n_segments, p=[0.7, 0.3])

# ======================================================================
# Code Block 13
# ======================================================================

water_cut = rng.beta(2, 10, n_segments)  # Typically low but some wells have high water

# ======================================================================
# Code Block 14
# ======================================================================

avg_temp = 0.5 * (temp_in_celsius + temp_out_celsius)
thermal_margin = wat_celsius - avg_temp

# ======================================================================
# Code Block 15
# ======================================================================

base_logit = (
    0.8 * (thermal_margin > 0) +      # Below WAT
    0.4 * (thermal_margin > 2) +      # Well below WAT
    0.3 * (water_cut > 0.2) +         # High water content
    0.3 * (crude == 'heavy_waxy') -   # Waxy crude
    0.4 * inhibitor_active -           # Chemical protection
    0.3 * (shear_proxy > 0.9)         # High shear prevents deposition
)

# ======================================================================
# Code Block 16
# ======================================================================

risk_probability = 1 / (1 + np.exp(-(base_logit - 0.4)))

# ======================================================================
# Code Block 17
# ======================================================================

risk_observed = (rng.random(n_segments) < risk_probability).astype(int)

# ======================================================================
# Code Block 18
# ======================================================================

telemetry = pd.DataFrame({
    'crude_type': crude,
    'wat_celsius': wat_celsius,
    'temp_in_celsius': temp_in_celsius,
    'temp_out_celsius': temp_out_celsius,
    'pressure_bar': pressure_bar,
    'flow_ksm3h': flow_ksm3h,
    'shear_proxy': shear_proxy,
    'inhibitor_active': inhibitor_active,
    'water_cut': water_cut,
    'risk_observed': risk_observed
})

return telemetry

# ======================================================================
# Code Block 19
# ======================================================================

X = telemetry_data.drop(columns=['risk_observed'])
y = telemetry_data['risk_observed']

# ======================================================================
# Code Block 20
# ======================================================================

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = ['crude_type']

# ======================================================================
# Code Block 21
# ======================================================================

from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# ======================================================================
# Code Block 22
# ======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

# ======================================================================
# Code Block 23
# ======================================================================

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=3,
        n_jobs=-1
    ))
])

# ======================================================================
# Code Block 24
# ======================================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ======================================================================
# Code Block 25
# ======================================================================

roc_auc = roc_auc_score(y_test, y_proba)

# ======================================================================
# Code Block 26
# ======================================================================

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ======================================================================
# Code Block 27
# ======================================================================

sensitivity = tp / max(1, tp + fn)  # True positive rate
specificity = tn / max(1, tn + fp)  # True negative rate
precision = tp / max(1, tp + fp)    # Positive predictive value
npv = tn / max(1, tn + fn)          # Negative predictive value

# ======================================================================
# Code Block 28
# ======================================================================

class_report = classification_report(y_test, y_pred, output_dict=True)

return {
    'model': model,
    'X_test': X_test,
    'y_test': y_test,
    'y_pred': y_pred,
    'y_proba': y_proba,
    'roc_auc': roc_auc,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'precision': precision,
    'npv': npv,
    'confusion_matrix': cm,
    'classification_report': class_report
}

# ======================================================================
# Code Block 29
# ======================================================================

test_segments = model_results['X_test'].copy()
test_segments['risk_probability'] = model_results['y_proba']
test_segments['actual_risk'] = model_results['y_test'].values

# ======================================================================
# Code Block 30
# ======================================================================

test_segments['criticality_score'] = (
    test_segments['pressure_bar'] * 0.6 +
    test_segments['flow_ksm3h'] * 10.0  # Scale flow to comparable range
)

# ======================================================================
# Code Block 31
# ======================================================================

test_segments['risk_rank_score'] = (
    test_segments['risk_probability'] * 
    (1 + 0.003 * test_segments['criticality_score'])
)

# ======================================================================
# Code Block 32
# ======================================================================

test_segments['thermal_margin'] = (
    test_segments['wat_celsius'] - 
    0.5 * (test_segments['temp_in_celsius'] + test_segments['temp_out_celsius'])
)

# ======================================================================
# Code Block 33
# ======================================================================

test_segments['risk_category'] = pd.cut(
    test_segments['risk_probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['LOW', 'MEDIUM', 'HIGH']
)

# ======================================================================
# Code Block 34
# ======================================================================

action_map = {
    'LOW': 'Continue monitoring',
    'MEDIUM': 'Increase inspection frequency',
    'HIGH': 'Priority intervention - consider pigging or inhibitor boost'
}
test_segments['recommended_action'] = test_segments['risk_category'].map(action_map)

# ======================================================================
# Code Block 35
# ======================================================================

top_segments = test_segments.sort_values('risk_rank_score', ascending=False).head(top_n)

# ======================================================================
# Code Block 36
# ======================================================================

report_columns = [
    'risk_probability', 'risk_rank_score', 'risk_category',
    'crude_type', 'wat_celsius', 'thermal_margin',
    'temp_in_celsius', 'temp_out_celsius',
    'pressure_bar', 'flow_ksm3h',
    'inhibitor_active', 'water_cut',
    'recommended_action'
]

return top_segments[report_columns]

# ======================================================================
# Code Block 37
# ======================================================================

print(f"\nSegment {idx} (Risk Score: {row['risk_rank_score']:.3f}):")
print(f"  Risk Probability: {row['risk_probability']:.1%} ({row['risk_category']})")
print(f"  Crude Type: {row['crude_type']}")
print(f"  WAT: {row['wat_celsius']:.1f}°C, Thermal Margin: {row['thermal_margin']:+.1f}°C")
print(f"  Temp In: {row['temp_in_celsius']:.1f}°C, Temp Out: {row['temp_out_celsius']:.1f}°C")
print(f"  Pressure: {row['pressure_bar']:.1f} bar, Flow: {row['flow_ksm3h']:.2f} kSm³/h")
print(f"  Inhibitor: {'Active' if row['inhibitor_active'] else 'Inactive'}")
print(f"  Water Cut: {row['water_cut']:.1%}")
print(f"  Recommendation: {row['recommended_action']}")

# ======================================================================
# Code Block 38
# ======================================================================

rf_model = model_results['model'].named_steps['classifier']

# ======================================================================
# Code Block 39
# ======================================================================

importances = rf_model.feature_importances_

# ======================================================================
# Code Block 40
# ======================================================================

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# ======================================================================
# Code Block 41
# ======================================================================

importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

return importance_df

# ======================================================================
# Code Block 42
# ======================================================================

print(f"{row['feature']:20s}: {row['importance']:.3f} ({row['cumulative_importance']:.1%} cumulative)")
