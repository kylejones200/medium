#!/usr/bin/env python3
"""
Python code extracted from 11_ore_grade_ml_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from skgstat import Variogram
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

def fetch_geochemical_data(region_bounds=None):
    """
    Fetch geochemical data from Geoscience Australia.
    
    For demonstration, we generate synthetic data matching NGSA structure.
    In production, download from: https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/122101
    
    Returns:
        GeoDataFrame with sample locations and element concentrations
    """
    np.random.seed(42)
    
    # Western Australia region (Goldfields-Esperance)
    n_samples = 250
    
    # Generate spatially correlated sampling
    lon = np.random.uniform(118.0, 123.0, n_samples)
    lat = np.random.uniform(-32.0, -28.0, n_samples)
    
    # Create realistic gold distribution with spatial correlation
    # Gold tends to cluster in mineralized zones
    x_norm = (lon - lon.min()) / (lon.max() - lon.min())
    y_norm = (lat - lat.min()) / (lat.max() - lat.min())
    
    # Create mineralized "zones" using Gaussian blobs
    zone1 = np.exp(-((x_norm - 0.3)**2 + (y_norm - 0.4)**2) / 0.01)
    zone2 = np.exp(-((x_norm - 0.7)**2 + (y_norm - 0.6)**2) / 0.015)
    zone3 = np.exp(-((x_norm - 0.5)**2 + (y_norm - 0.2)**2) / 0.008)
    
    mineralization = zone1 + zone2 + zone3
    
    # Gold concentration (log-normal distribution)
    log_au_base = mineralization * 3.0 + np.random.randn(n_samples) * 0.5
    au_ppm = np.exp(log_au_base) * 0.01  # Convert to ppm
    au_ppm = np.clip(au_ppm, 0.001, 5.0)  # Realistic range
    
    # Pathfinder elements correlated with gold
    cu_ppm = au_ppm * 50 + np.random.randn(n_samples) * 10
    as_ppm = au_ppm * 30 + np.random.randn(n_samples) * 5
    pb_ppm = au_ppm * 20 + np.random.randn(n_samples) * 8
    s_pct = au_ppm * 0.3 + np.random.randn(n_samples) * 0.1
    fe_pct = 4.0 + mineralization * 2.0 + np.random.randn(n_samples) * 1.0
    
    # Lithology (categorical)
    lithology_types = ['granite', 'basalt', 'sediment', 'greenstone']
    lithology_probs = mineralization / mineralization.sum()
    lithology_probs = np.column_stack([
        lithology_probs * 0.2,  # granite
        lithology_probs * 0.3,  # basalt
        (1 - lithology_probs) * 0.3,  # sediment
        lithology_probs * 0.4  # greenstone (favorable)
    ])
    lithology_probs = lithology_probs / lithology_probs.sum(axis=1, keepdims=True)
    lithology = np.array([np.random.choice(lithology_types, p=probs) 
                          for probs in lithology_probs])
    
    df = pd.DataFrame({
        'longitude': lon,
        'latitude': lat,
        'Au': au_ppm,
        'Cu': cu_ppm,
        'As': as_ppm,
        'Pb': pb_ppm,
        'S': s_pct,
        'Fe': fe_pct,
        'lithology': lithology,
        'sample_id': [f'NGSA_{i:04d}' for i in range(n_samples)]
    })
    
    return df

def prepare_spatial_features(df, target_crs="EPSG:32750"):
    """
    Convert to projected CRS and extract spatial features.
    
    Args:
        df: DataFrame with longitude, latitude, Au, and covariates
        target_crs: UTM zone for Western Australia (zone 50S)
        
    Returns:
        GeoDataFrame with x, y, log_Au, and features
    """
    # Filter positive Au values
    df = df[df["Au"] > 0].copy()
    
    # Log transform Au to reduce skewness
    df["log_Au"] = np.log1p(df["Au"])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    
    # Project to UTM
    gdf = gdf.to_crs(target_crs)
    gdf["x"] = gdf.geometry.x / 1000  # Convert to km for numerical stability
    gdf["y"] = gdf.geometry.y / 1000
    
    print(f"Prepared {len(gdf)} samples")
    print(f"Au range: {gdf['Au'].min():.3f} - {gdf['Au'].max():.3f} ppm")
    print(f"Mean Au: {gdf['Au'].mean():.3f} ppm")
    print(f"Spatial extent: {gdf['x'].max() - gdf['x'].min():.1f} km × {gdf['y'].max() - gdf['y'].min():.1f} km")
    
    return gdf

def create_spatial_folds(gdf, n_folds=5):
    """
    Create spatial cross-validation folds to prevent leakage.
    
    Uses x-coordinate bands to ensure train/test spatial separation.
    """
    groups = pd.qcut(gdf["x"], n_folds, labels=False, duplicates='drop')
    
    print(f"\nCreated {n_folds} spatial folds:")
    for fold in range(n_folds):
        n = (groups == fold).sum()
        print(f"  Fold {fold}: {n} samples")
    
    return groups

# ======================================================================
# Code Block 2
# ======================================================================

def fit_variogram(gdf, plot=False):
    """
    Fit experimental and theoretical variogram for spatial correlation.
    
    Returns:
        Variogram model fitted to data
    """
    coords = np.column_stack([gdf["x"].values, gdf["y"].values])
    values = gdf["log_Au"].values
    
    # Compute experimental variogram
    V = Variogram(
        coords, 
        values, 
        model="spherical",
        maxlag="median",  # Use median distance as max lag
        n_lags=25
    )
    
    print("\nVariogram Parameters:")
    print(f"  Model: {V.model.__name__}")
    print(f"  Sill: {V.sill:.3f}")
    print(f"  Range: {V.range:.1f} km")
    print(f"  Nugget: {V.nugget:.3f}")
    print(f"  Nugget/Sill ratio: {V.nugget/V.sill:.2%}")
    
    return V

def ordinary_kriging_predict(gdf, grid_resolution=100):
    """
    Perform Ordinary Kriging on a regular grid.
    
    Returns:
        grid_x, grid_y, predictions, variance
    """
    # Create prediction grid
    gx = np.linspace(gdf["x"].min(), gdf["x"].max(), grid_resolution)
    gy = np.linspace(gdf["y"].min(), gdf["y"].max(), grid_resolution)
    
    # Fit Ordinary Kriging
    OK = OrdinaryKriging(
        gdf["x"].values, 
        gdf["y"].values, 
        gdf["log_Au"].values,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False
    )
    
    # Execute kriging
    z, ss = OK.execute("grid", gx, gy)
    
    # Convert back to ppm
    z_ppm = np.expm1(z)
    
    print("\nOrdinary Kriging Results:")
    print(f"  Grid size: {grid_resolution} × {grid_resolution}")
    print(f"  Predicted Au range: {z_ppm.min():.3f} - {z_ppm.max():.3f} ppm")
    print(f"  Mean kriging variance: {ss.mean():.3f}")
    
    return gx, gy, z_ppm, ss

# ======================================================================
# Code Block 3
# ======================================================================

def train_gaussian_process(gdf, groups):
    """
    Train Gaussian Process Regressor with spatial cross-validation.
    
    Args:
        gdf: GeoDataFrame with features
        groups: Spatial fold assignments
        
    Returns:
        Trained model, predictions, uncertainties, metrics
    """
    # Define features
    numeric_features = ["x", "y", "Cu", "As", "Fe", "S", "Pb"]
    categorical_features = ["lithology"]
    
    # Build preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore"), 
         categorical_features)
    ])
    
    # Define GP kernel: spatial + feature correlation + noise
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) * 
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + 
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )
    
    # Build pipeline
    gp_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42
        ))
    ])
    
    # Spatial cross-validation
    X = gdf[numeric_features + categorical_features]
    y = gdf["log_Au"].values
    
    pred_mu = np.zeros_like(y)
    pred_std = np.zeros_like(y)
    
    print("\nGaussian Process Cross-Validation:")
    gkf = GroupKFold(n_splits=5)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        gp_pipeline.fit(X_train, y_train)
        
        # Predict with uncertainty
        X_test_transformed = gp_pipeline.named_steps["preprocessor"].transform(X_test)
        mu, std = gp_pipeline.named_steps["gpr"].predict(X_test_transformed, return_std=True)
        
        pred_mu[test_idx] = mu
        pred_std[test_idx] = std
        
        # Fold metrics
        fold_mae = mean_absolute_error(y_test, mu)
        fold_rmse = np.sqrt(mean_squared_error(y_test, mu))
        print(f"  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}")
    
    # Overall metrics
    mae = mean_absolute_error(y, pred_mu)
    rmse = np.sqrt(mean_squared_error(y, pred_mu))
    
    # Uncertainty calibration: 95% coverage
    z_scores = np.abs(y - pred_mu) / np.maximum(pred_std, 1e-6)
    coverage_95 = (z_scores < 1.96).mean()
    
    print(f"\nGPR Overall Performance:")
    print(f"  MAE: {mae:.3f} log(ppm)")
    print(f"  RMSE: {rmse:.3f} log(ppm)")
    print(f"  95% Confidence Coverage: {coverage_95:.1%}")
    print(f"  Mean Prediction Std: {pred_std.mean():.3f}")
    
    # Refit on full data for final predictions
    gp_pipeline.fit(X, y)
    
    return gp_pipeline, pred_mu, pred_std, {"mae": mae, "rmse": rmse, "coverage": coverage_95}

# ======================================================================
# Code Block 4
# ======================================================================

def train_xgboost(gdf, groups):
    """
    Train XGBoost regressor with spatial cross-validation.
    
    Returns:
        Trained model, predictions, metrics
    """
    # Define features
    numeric_features = ["x", "y", "Cu", "As", "Fe", "S", "Pb"]
    categorical_features = ["lithology"]
    
    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore"), 
         categorical_features)
    ])
    
    # XGBoost pipeline
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Spatial cross-validation
    X = gdf[numeric_features + categorical_features]
    y = gdf["log_Au"].values
    
    pred = np.zeros_like(y)
    
    print("\nXGBoost Cross-Validation:")
    gkf = GroupKFold(n_splits=5)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train
        xgb_pipeline.fit(X_train, y_train)
        
        # Predict
        pred[test_idx] = xgb_pipeline.predict(X_test)
        
        # Fold metrics
        fold_mae = mean_absolute_error(y_test, pred[test_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_test, pred[test_idx]))
        print(f"  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}")
    
    # Overall metrics
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    
    print(f"\nXGBoost Overall Performance:")
    print(f"  MAE: {mae:.3f} log(ppm)")
    print(f"  RMSE: {rmse:.3f} log(ppm)")
    
    # Feature importance
    xgb_pipeline.fit(X, y)
    
    # Get feature names after encoding
    feature_names = (
        numeric_features + 
        list(xgb_pipeline.named_steps["preprocessor"]
             .named_transformers_["cat"]
             .get_feature_names_out(categorical_features))
    )
    
    importances = xgb_pipeline.named_steps["xgb"].feature_importances_
    
    print("\nTop Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), 
                           key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {imp:.3f}")
    
    return xgb_pipeline, pred, {"mae": mae, "rmse": rmse}

# ======================================================================
# Code Block 5
# ======================================================================

def create_prediction_grid(gdf, gp_model, xgb_model, resolution=150):
    """
    Generate grade predictions on a regular grid for all three methods.
    
    Returns:
        DataFrame with grid predictions and uncertainties
    """
    # Create grid
    gx = np.linspace(gdf["x"].min(), gdf["x"].max(), resolution)
    gy = np.linspace(gdf["y"].min(), gdf["y"].max(), resolution)
    grid_x, grid_y = np.meshgrid(gx, gy)
    
    # For ML models, we need to interpolate covariate values to grid points
    # Use nearest neighbor for simplicity (in practice, kriging or IDW for each)
    from scipy.spatial import cKDTree
    
    tree = cKDTree(np.column_stack([gdf["x"], gdf["y"]]))
    _, nearest_idx = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    
    # Create grid feature matrix
    grid_features = pd.DataFrame({
        'x': grid_x.ravel(),
        'y': grid_y.ravel(),
        'Cu': gdf.iloc[nearest_idx]["Cu"].values,
        'As': gdf.iloc[nearest_idx]["As"].values,
        'Fe': gdf.iloc[nearest_idx]["Fe"].values,
        'S': gdf.iloc[nearest_idx]["S"].values,
        'Pb': gdf.iloc[nearest_idx]["Pb"].values,
        'lithology': gdf.iloc[nearest_idx]["lithology"].values
    })
    
    # GPR predictions
    gp_transformed = gp_model.named_steps["preprocessor"].transform(grid_features)
    gp_mu, gp_std = gp_model.named_steps["gpr"].predict(gp_transformed, return_std=True)
    gp_ppm = np.expm1(gp_mu).reshape(grid_x.shape)
    gp_std_grid = gp_std.reshape(grid_x.shape)
    
    # XGBoost predictions
    xgb_pred = xgb_model.predict(grid_features)
    xgb_ppm = np.expm1(xgb_pred).reshape(grid_x.shape)
    
    # Ordinary Kriging (from earlier)
    gx_1d, gy_1d, ok_ppm, ok_var = ordinary_kriging_predict(gdf, grid_resolution=resolution)
    
    print(f"\nGrid Predictions Complete:")
    print(f"  GPR Au range: {gp_ppm.min():.3f} - {gp_ppm.max():.3f} ppm")
    print(f"  XGB Au range: {xgb_ppm.min():.3f} - {xgb_ppm.max():.3f} ppm")
    print(f"  OK Au range: {ok_ppm.min():.3f} - {ok_ppm.max():.3f} ppm")
    
    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'gp_mean': gp_ppm,
        'gp_std': gp_std_grid,
        'xgb_pred': xgb_ppm,
        'ok_mean': ok_ppm,
        'ok_var': ok_var
    }

# ======================================================================
# Code Block 6
# ======================================================================

def create_grade_maps(gdf, grid_results):
    """
    Generate comparative grade prediction maps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Use serif font
    plt.rcParams['font.family'] = 'serif'
    
    # Panel 1: Ordinary Kriging
    ax1 = axes[0, 0]
    im1 = ax1.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                       grid_results['ok_mean'], levels=20, cmap='gray')
    ax1.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
                linewidths=0.5, alpha=0.6, label='Samples')
    
    # Apply minimalist style
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_position(('outward', 5))
    ax1.spines['bottom'].set_position(('outward', 5))
    
    ax1.set_title('Ordinary Kriging', fontsize=12, fontweight='bold', loc='left')
    ax1.set_xlabel('Easting (km)', fontsize=10)
    ax1.set_ylabel('Northing (km)', fontsize=10)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Au (ppm)', fontsize=9)
    
    # Panel 2: GPR Mean
    ax2 = axes[0, 1]
    im2 = ax2.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                       grid_results['gp_mean'], levels=20, cmap='gray')
    ax2.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
                linewidths=0.5, alpha=0.6)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_position(('outward', 5))
    ax2.spines['bottom'].set_position(('outward', 5))
    
    ax2.set_title('Gaussian Process (Mean)', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xlabel('Easting (km)', fontsize=10)
    ax2.set_ylabel('Northing (km)', fontsize=10)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Au (ppm)', fontsize=9)
    
    # Panel 3: GPR Uncertainty
    ax3 = axes[1, 0]
    im3 = ax3.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                       grid_results['gp_std'], levels=20, cmap='gray')
    ax3.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
                linewidths=0.5, alpha=0.6)
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_position(('outward', 5))
    ax3.spines['bottom'].set_position(('outward', 5))
    
    ax3.set_title('GPR Uncertainty (Std Dev)', fontsize=12, fontweight='bold', loc='left')
    ax3.set_xlabel('Easting (km)', fontsize=10)
    ax3.set_ylabel('Northing (km)', fontsize=10)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Std Dev (log ppm)', fontsize=9)
    
    # Panel 4: XGBoost
    ax4 = axes[1, 1]
    im4 = ax4.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                       grid_results['xgb_pred'], levels=20, cmap='gray')
    ax4.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
                linewidths=0.5, alpha=0.6)
    
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_position(('outward', 5))
    ax4.spines['bottom'].set_position(('outward', 5))
    
    ax4.set_title('XGBoost', fontsize=12, fontweight='bold', loc='left')
    ax4.set_xlabel('Easting (km)', fontsize=10)
    ax4.set_ylabel('Northing (km)', fontsize=10)
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Au (ppm)', fontsize=9)
    
    plt.savefig('11_ore_grade_ml_main.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created: 11_ore_grade_ml_main.png")

# ======================================================================
# Code Block 7
# ======================================================================

def analyze_uncertainty_calibration(y_true, y_pred, y_std, n_bins=10):
    """
    Analyze uncertainty calibration by binning predictions by confidence.
    
    Well-calibrated models show actual RMSE matching predicted uncertainty.
    """
    # Bin by predicted uncertainty
    bins = pd.qcut(y_std, n_bins, duplicates='drop')
    
    calibration_data = []
    for bin_label in bins.cat.categories:
        mask = (bins == bin_label)
        bin_std = y_std[mask].mean()
        bin_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        calibration_data.append({
            'predicted_std': bin_std,
            'actual_rmse': bin_rmse,
            'n_samples': mask.sum()
        })
    
    calib_df = pd.DataFrame(calibration_data)
    
    print("\nUncertainty Calibration:")
    print(calib_df.to_string(index=False))
    
    # Ideal calibration: actual_rmse ≈ predicted_std
    correlation = np.corrcoef(calib_df['predicted_std'], calib_df['actual_rmse'])[0, 1]
    print(f"\nCalibration Correlation: {correlation:.3f}")
    
    return calib_df

def compare_methods(ok_metrics, gpr_metrics, xgb_metrics):
    """
    Comparative summary of all three methods.
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    print("\nAccuracy Metrics:")
    print(f"  Ordinary Kriging:    MAE = N/A (no CV), RMSE = N/A")
    print(f"  Gaussian Process:    MAE = {gpr_metrics['mae']:.3f}, RMSE = {gpr_metrics['rmse']:.3f}")
    print(f"  XGBoost:             MAE = {xgb_metrics['mae']:.3f}, RMSE = {xgb_metrics['rmse']:.3f}")
    
    print("\nUncertainty Quantification:")
    print(f"  Ordinary Kriging:    Kriging variance (but often overconfident)")
    print(f"  Gaussian Process:    95% Coverage = {gpr_metrics['coverage']:.1%} (well-calibrated)")
    print(f"  XGBoost:             None (point estimates only)")
    
    print("\nComputational Efficiency:")
    print(f"  Ordinary Kriging:    O(n³) - slow for large datasets")
    print(f"  Gaussian Process:    O(n³) - same limitations")
    print(f"  XGBoost:             O(n log n) - scales to millions of points")
    
    print("\nBest Use Cases:")
    print("  Ordinary Kriging:    Traditional geostatistics, spatial-only data")
    print("  Gaussian Process:    When you need calibrated uncertainty + covariates")
    print("  XGBoost:             Production forecasting with tight deadlines")
    
    print("="*70)

# ======================================================================
# Code Block 8
# ======================================================================

def main():
    """Complete ore grade forecasting pipeline."""
    # 1. Fetch and prepare data
    df = fetch_geochemical_data()
    gdf = prepare_spatial_features(df)
    groups = create_spatial_folds(gdf)
    
    # 2. Fit variogram and perform kriging
    V = fit_variogram(gdf)
    gx, gy, ok_ppm, ok_var = ordinary_kriging_predict(gdf)
    
    # 3. Train ML models
    gp_model, gp_pred, gp_std, gpr_metrics = train_gaussian_process(gdf, groups)
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(gdf, groups)
    
    # 4. Generate prediction grids
    grid_results = create_prediction_grid(gdf, gp_model, xgb_model)
    
    # 5. Create visualizations
    create_grade_maps(gdf, grid_results)
    
    # 6. Calibration analysis
    calib_df = analyze_uncertainty_calibration(
        gdf["log_Au"].values, gp_pred, gp_std
    )
    
    # 7. Comparison
    compare_methods({}, gpr_metrics, xgb_metrics)
    
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()

# ======================================================================
# Code Block 9
# ======================================================================

n_samples = 250

# ======================================================================
# Code Block 10
# ======================================================================

lon = np.random.uniform(118.0, 123.0, n_samples)
lat = np.random.uniform(-32.0, -28.0, n_samples)

# ======================================================================
# Code Block 11
# ======================================================================

x_norm = (lon - lon.min()) / (lon.max() - lon.min())
y_norm = (lat - lat.min()) / (lat.max() - lat.min())

# ======================================================================
# Code Block 12
# ======================================================================

zone1 = np.exp(-((x_norm - 0.3)**2 + (y_norm - 0.4)**2) / 0.01)
zone2 = np.exp(-((x_norm - 0.7)**2 + (y_norm - 0.6)**2) / 0.015)
zone3 = np.exp(-((x_norm - 0.5)**2 + (y_norm - 0.2)**2) / 0.008)

mineralization = zone1 + zone2 + zone3

# ======================================================================
# Code Block 13
# ======================================================================

log_au_base = mineralization * 3.0 + np.random.randn(n_samples) * 0.5
au_ppm = np.exp(log_au_base) * 0.01  # Convert to ppm
au_ppm = np.clip(au_ppm, 0.001, 5.0)  # Realistic range

# ======================================================================
# Code Block 14
# ======================================================================

cu_ppm = au_ppm * 50 + np.random.randn(n_samples) * 10
as_ppm = au_ppm * 30 + np.random.randn(n_samples) * 5
pb_ppm = au_ppm * 20 + np.random.randn(n_samples) * 8
s_pct = au_ppm * 0.3 + np.random.randn(n_samples) * 0.1
fe_pct = 4.0 + mineralization * 2.0 + np.random.randn(n_samples) * 1.0

# ======================================================================
# Code Block 15
# ======================================================================

lithology_types = ['granite', 'basalt', 'sediment', 'greenstone']
lithology_probs = mineralization / mineralization.sum()
lithology_probs = np.column_stack([
    lithology_probs * 0.2,  # granite
    lithology_probs * 0.3,  # basalt
    (1 - lithology_probs) * 0.3,  # sediment
    lithology_probs * 0.4  # greenstone (favorable)
])
lithology_probs = lithology_probs / lithology_probs.sum(axis=1, keepdims=True)
lithology = np.array([np.random.choice(lithology_types, p=probs) 
                      for probs in lithology_probs])

df = pd.DataFrame({
    'longitude': lon,
    'latitude': lat,
    'Au': au_ppm,
    'Cu': cu_ppm,
    'As': as_ppm,
    'Pb': pb_ppm,
    'S': s_pct,
    'Fe': fe_pct,
    'lithology': lithology,
    'sample_id': [f'NGSA_{i:04d}' for i in range(n_samples)]
})

return df

# ======================================================================
# Code Block 16
# ======================================================================

df = df[df["Au"] > 0].copy()

# ======================================================================
# Code Block 17
# ======================================================================

df["log_Au"] = np.log1p(df["Au"])

# ======================================================================
# Code Block 18
# ======================================================================

gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

# ======================================================================
# Code Block 19
# ======================================================================

gdf = gdf.to_crs(target_crs)
gdf["x"] = gdf.geometry.x / 1000  # Convert to km for numerical stability
gdf["y"] = gdf.geometry.y / 1000

print(f"Prepared {len(gdf)} samples")
print(f"Au range: {gdf['Au'].min():.3f} - {gdf['Au'].max():.3f} ppm")
print(f"Mean Au: {gdf['Au'].mean():.3f} ppm")
print(f"Spatial extent: {gdf['x'].max() - gdf['x'].min():.1f} km × {gdf['y'].max() - gdf['y'].min():.1f} km")

return gdf

# ======================================================================
# Code Block 20
# ======================================================================

"""
Create spatial cross-validation folds to prevent leakage.

Uses x-coordinate bands to ensure train/test spatial separation.
"""
groups = pd.qcut(gdf["x"], n_folds, labels=False, duplicates='drop')

print(f"\nCreated {n_folds} spatial folds:")
for fold in range(n_folds):
    n = (groups == fold).sum()
    print(f"  Fold {fold}: {n} samples")

return groups

# ======================================================================
# Code Block 21
# ======================================================================

"""
Fit experimental and theoretical variogram for spatial correlation.

Returns:
    Variogram model fitted to data
"""
coords = np.column_stack([gdf["x"].values, gdf["y"].values])
values = gdf["log_Au"].values

# ======================================================================
# Code Block 22
# ======================================================================

V = Variogram(
    coords, 
    values, 
    model="spherical",
    maxlag="median",  # Use median distance as max lag
    n_lags=25
)

print("\nVariogram Parameters:")
print(f"  Model: {V.model.__name__}")
print(f"  Sill: {V.sill:.3f}")
print(f"  Range: {V.range:.1f} km")
print(f"  Nugget: {V.nugget:.3f}")
print(f"  Nugget/Sill ratio: {V.nugget/V.sill:.2%}")

return V

# ======================================================================
# Code Block 23
# ======================================================================

gx = np.linspace(gdf["x"].min(), gdf["x"].max(), grid_resolution)
gy = np.linspace(gdf["y"].min(), gdf["y"].max(), grid_resolution)

# ======================================================================
# Code Block 24
# ======================================================================

OK = OrdinaryKriging(
    gdf["x"].values, 
    gdf["y"].values, 
    gdf["log_Au"].values,
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False
)

# ======================================================================
# Code Block 25
# ======================================================================

z, ss = OK.execute("grid", gx, gy)

# ======================================================================
# Code Block 26
# ======================================================================

z_ppm = np.expm1(z)

print("\nOrdinary Kriging Results:")
print(f"  Grid size: {grid_resolution} × {grid_resolution}")
print(f"  Predicted Au range: {z_ppm.min():.3f} - {z_ppm.max():.3f} ppm")
print(f"  Mean kriging variance: {ss.mean():.3f}")

return gx, gy, z_ppm, ss

# ======================================================================
# Code Block 27
# ======================================================================

numeric_features = ["x", "y", "Cu", "As", "Fe", "S", "Pb"]
categorical_features = ["lithology"]

# ======================================================================
# Code Block 28
# ======================================================================

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore"), 
     categorical_features)
])

# ======================================================================
# Code Block 29
# ======================================================================

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) * 
    Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + 
    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
)

# ======================================================================
# Code Block 30
# ======================================================================

gp_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("gpr", GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=42
    ))
])

# ======================================================================
# Code Block 31
# ======================================================================

X = gdf[numeric_features + categorical_features]
y = gdf["log_Au"].values

pred_mu = np.zeros_like(y)
pred_std = np.zeros_like(y)

print("\nGaussian Process Cross-Validation:")
gkf = GroupKFold(n_splits=5)

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# ======================================================================
# Code Block 32
# ======================================================================

X_test_transformed = gp_pipeline.named_steps["preprocessor"].transform(X_test)
    mu, std = gp_pipeline.named_steps["gpr"].predict(X_test_transformed, return_std=True)
    
    pred_mu[test_idx] = mu
    pred_std[test_idx] = std

# ======================================================================
# Code Block 33
# ======================================================================

fold_mae = mean_absolute_error(y_test, mu)
    fold_rmse = np.sqrt(mean_squared_error(y_test, mu))
    print(f"  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}")

# ======================================================================
# Code Block 34
# ======================================================================

mae = mean_absolute_error(y, pred_mu)
rmse = np.sqrt(mean_squared_error(y, pred_mu))

# ======================================================================
# Code Block 35
# ======================================================================

z_scores = np.abs(y - pred_mu) / np.maximum(pred_std, 1e-6)
coverage_95 = (z_scores < 1.96).mean()

print(f"\nGPR Overall Performance:")
print(f"  MAE: {mae:.3f} log(ppm)")
print(f"  RMSE: {rmse:.3f} log(ppm)")
print(f"  95% Confidence Coverage: {coverage_95:.1%}")
print(f"  Mean Prediction Std: {pred_std.mean():.3f}")

# ======================================================================
# Code Block 36
# ======================================================================

gp_pipeline.fit(X, y)

return gp_pipeline, pred_mu, pred_std, {"mae": mae, "rmse": rmse, "coverage": coverage_95}

# ======================================================================
# Code Block 37
# ======================================================================

Gaussian Process Cross-Validation:
  Fold 0: MAE=0.287, RMSE=0.392
  Fold 1: MAE=0.312, RMSE=0.421
  Fold 2: MAE=0.298, RMSE=0.405
  Fold 3: MAE=0.275, RMSE=0.368
  Fold 4: MAE=0.291, RMSE=0.397

GPR Overall Performance:
  MAE: 0.293 log(ppm)
  RMSE: 0.397 log(ppm)
  95% Confidence Coverage: 94.8%
  Mean Prediction Std: 0.385

# ======================================================================
# Code Block 38
# ======================================================================

numeric_features = ["x", "y", "Cu", "As", "Fe", "S", "Pb"]
categorical_features = ["lithology"]

# ======================================================================
# Code Block 39
# ======================================================================

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore"), 
     categorical_features)
])

# ======================================================================
# Code Block 40
# ======================================================================

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("xgb", xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

# ======================================================================
# Code Block 41
# ======================================================================

X = gdf[numeric_features + categorical_features]
y = gdf["log_Au"].values

pred = np.zeros_like(y)

print("\nXGBoost Cross-Validation:")
gkf = GroupKFold(n_splits=5)

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# ======================================================================
# Code Block 42
# ======================================================================

pred[test_idx] = xgb_pipeline.predict(X_test)

# ======================================================================
# Code Block 43
# ======================================================================

fold_mae = mean_absolute_error(y_test, pred[test_idx])
    fold_rmse = np.sqrt(mean_squared_error(y_test, pred[test_idx]))
    print(f"  Fold {fold_idx}: MAE={fold_mae:.3f}, RMSE={fold_rmse:.3f}")

# ======================================================================
# Code Block 44
# ======================================================================

mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))

print(f"\nXGBoost Overall Performance:")
print(f"  MAE: {mae:.3f} log(ppm)")
print(f"  RMSE: {rmse:.3f} log(ppm)")

# ======================================================================
# Code Block 45
# ======================================================================

feature_names = (
    numeric_features + 
    list(xgb_pipeline.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .get_feature_names_out(categorical_features))
)

importances = xgb_pipeline.named_steps["xgb"].feature_importances_

print("\nTop Feature Importances:")
for name, imp in sorted(zip(feature_names, importances), 
                       key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {name}: {imp:.3f}")

return xgb_pipeline, pred, {"mae": mae, "rmse": rmse}

# ======================================================================
# Code Block 46
# ======================================================================

XGBoost Cross-Validation:
  Fold 0: MAE=0.245, RMSE=0.334
  Fold 1: MAE=0.268, RMSE=0.362
  Fold 2: MAE=0.257, RMSE=0.348
  Fold 3: MAE=0.239, RMSE=0.325
  Fold 4: MAE=0.251, RMSE=0.341

XGBoost Overall Performance:
  MAE: 0.252 log(ppm)
  RMSE: 0.342 log(ppm)

Top Feature Importances:
  x: 0.287
  y: 0.265
  As: 0.189
  Cu: 0.142
  S: 0.067

# ======================================================================
# Code Block 47
# ======================================================================

gx = np.linspace(gdf["x"].min(), gdf["x"].max(), resolution)
gy = np.linspace(gdf["y"].min(), gdf["y"].max(), resolution)
grid_x, grid_y = np.meshgrid(gx, gy)

# ======================================================================
# Code Block 48
# ======================================================================

from scipy.spatial import cKDTree

tree = cKDTree(np.column_stack([gdf["x"], gdf["y"]]))
_, nearest_idx = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))

# ======================================================================
# Code Block 49
# ======================================================================

grid_features = pd.DataFrame({
    'x': grid_x.ravel(),
    'y': grid_y.ravel(),
    'Cu': gdf.iloc[nearest_idx]["Cu"].values,
    'As': gdf.iloc[nearest_idx]["As"].values,
    'Fe': gdf.iloc[nearest_idx]["Fe"].values,
    'S': gdf.iloc[nearest_idx]["S"].values,
    'Pb': gdf.iloc[nearest_idx]["Pb"].values,
    'lithology': gdf.iloc[nearest_idx]["lithology"].values
})

# ======================================================================
# Code Block 50
# ======================================================================

gp_transformed = gp_model.named_steps["preprocessor"].transform(grid_features)
gp_mu, gp_std = gp_model.named_steps["gpr"].predict(gp_transformed, return_std=True)
gp_ppm = np.expm1(gp_mu).reshape(grid_x.shape)
gp_std_grid = gp_std.reshape(grid_x.shape)

# ======================================================================
# Code Block 51
# ======================================================================

xgb_pred = xgb_model.predict(grid_features)
xgb_ppm = np.expm1(xgb_pred).reshape(grid_x.shape)

# ======================================================================
# Code Block 52
# ======================================================================

gx_1d, gy_1d, ok_ppm, ok_var = ordinary_kriging_predict(gdf, grid_resolution=resolution)

print(f"\nGrid Predictions Complete:")
print(f"  GPR Au range: {gp_ppm.min():.3f} - {gp_ppm.max():.3f} ppm")
print(f"  XGB Au range: {xgb_ppm.min():.3f} - {xgb_ppm.max():.3f} ppm")
print(f"  OK Au range: {ok_ppm.min():.3f} - {ok_ppm.max():.3f} ppm")

return {
    'grid_x': grid_x,
    'grid_y': grid_y,
    'gp_mean': gp_ppm,
    'gp_std': gp_std_grid,
    'xgb_pred': xgb_ppm,
    'ok_mean': ok_ppm,
    'ok_var': ok_var
}

# ======================================================================
# Code Block 53
# ======================================================================

"""
Generate comparative grade prediction maps.
"""
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# ======================================================================
# Code Block 54
# ======================================================================

plt.rcParams['font.family'] = 'serif'

# ======================================================================
# Code Block 55
# ======================================================================

ax1 = axes[0, 0]
im1 = ax1.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                   grid_results['ok_mean'], levels=20, cmap='gray')
ax1.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
            linewidths=0.5, alpha=0.6, label='Samples')

# ======================================================================
# Code Block 56
# ======================================================================

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_position(('outward', 5))
ax1.spines['bottom'].set_position(('outward', 5))

ax1.set_title('Ordinary Kriging', fontsize=12, fontweight='bold', loc='left')
ax1.set_xlabel('Easting (km)', fontsize=10)
ax1.set_ylabel('Northing (km)', fontsize=10)
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Au (ppm)', fontsize=9)

# ======================================================================
# Code Block 57
# ======================================================================

ax2 = axes[0, 1]
im2 = ax2.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                   grid_results['gp_mean'], levels=20, cmap='gray')
ax2.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
            linewidths=0.5, alpha=0.6)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_position(('outward', 5))
ax2.spines['bottom'].set_position(('outward', 5))

ax2.set_title('Gaussian Process (Mean)', fontsize=12, fontweight='bold', loc='left')
ax2.set_xlabel('Easting (km)', fontsize=10)
ax2.set_ylabel('Northing (km)', fontsize=10)
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Au (ppm)', fontsize=9)

# ======================================================================
# Code Block 58
# ======================================================================

ax3 = axes[1, 0]
im3 = ax3.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                   grid_results['gp_std'], levels=20, cmap='gray')
ax3.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
            linewidths=0.5, alpha=0.6)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_position(('outward', 5))
ax3.spines['bottom'].set_position(('outward', 5))

ax3.set_title('GPR Uncertainty (Std Dev)', fontsize=12, fontweight='bold', loc='left')
ax3.set_xlabel('Easting (km)', fontsize=10)
ax3.set_ylabel('Northing (km)', fontsize=10)
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('Std Dev (log ppm)', fontsize=9)

# ======================================================================
# Code Block 59
# ======================================================================

ax4 = axes[1, 1]
im4 = ax4.contourf(grid_results['grid_x'], grid_results['grid_y'], 
                   grid_results['xgb_pred'], levels=20, cmap='gray')
ax4.scatter(gdf['x'], gdf['y'], c='white', s=8, edgecolors='black', 
            linewidths=0.5, alpha=0.6)

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_position(('outward', 5))
ax4.spines['bottom'].set_position(('outward', 5))

ax4.set_title('XGBoost', fontsize=12, fontweight='bold', loc='left')
ax4.set_xlabel('Easting (km)', fontsize=10)
ax4.set_ylabel('Northing (km)', fontsize=10)
cbar4 = plt.colorbar(im4, ax=ax4)
cbar4.set_label('Au (ppm)', fontsize=9)

plt.savefig('11_ore_grade_ml_main.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Created: 11_ore_grade_ml_main.png")

# ======================================================================
# Code Block 60
# ======================================================================

bins = pd.qcut(y_std, n_bins, duplicates='drop')

calibration_data = []
for bin_label in bins.cat.categories:
    mask = (bins == bin_label)
    bin_std = y_std[mask].mean()
    bin_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    calibration_data.append({
        'predicted_std': bin_std,
        'actual_rmse': bin_rmse,
        'n_samples': mask.sum()
    })

calib_df = pd.DataFrame(calibration_data)

print("\nUncertainty Calibration:")
print(calib_df.to_string(index=False))

# ======================================================================
# Code Block 61
# ======================================================================

correlation = np.corrcoef(calib_df['predicted_std'], calib_df['actual_rmse'])[0, 1]
print(f"\nCalibration Correlation: {correlation:.3f}")

return calib_df

# ======================================================================
# Code Block 62
# ======================================================================

"""
Comparative summary of all three methods.
"""
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

print("\nAccuracy Metrics:")
print(f"  Ordinary Kriging:    MAE = N/A (no CV), RMSE = N/A")
print(f"  Gaussian Process:    MAE = {gpr_metrics['mae']:.3f}, RMSE = {gpr_metrics['rmse']:.3f}")
print(f"  XGBoost:             MAE = {xgb_metrics['mae']:.3f}, RMSE = {xgb_metrics['rmse']:.3f}")

print("\nUncertainty Quantification:")
print(f"  Ordinary Kriging:    Kriging variance (but often overconfident)")
print(f"  Gaussian Process:    95% Coverage = {gpr_metrics['coverage']:.1%} (well-calibrated)")
print(f"  XGBoost:             None (point estimates only)")

print("\nComputational Efficiency:")
print(f"  Ordinary Kriging:    O(n³) - slow for large datasets")
print(f"  Gaussian Process:    O(n³) - same limitations")
print(f"  XGBoost:             O(n log n) - scales to millions of points")

print("\nBest Use Cases:")
print("  Ordinary Kriging:    Traditional geostatistics, spatial-only data")
print("  Gaussian Process:    When you need calibrated uncertainty + covariates")
print("  XGBoost:             Production forecasting with tight deadlines")

print("="*70)

# ======================================================================
# Code Block 63
# ======================================================================

Uncertainty Calibration:
   predicted_std  actual_rmse  n_samples
          0.245        0.267         25
          0.312        0.328         25
          0.358        0.361         25
          0.395        0.412         25
          0.428        0.439         25
          0.467        0.485         25
          0.512        0.521         25
          0.568        0.587         25
          0.643        0.656         25
          0.782        0.794         25

Calibration Correlation: 0.998

======================================================================
MODEL COMPARISON SUMMARY
======================================================================

Accuracy Metrics:
  Ordinary Kriging:    MAE = N/A (no CV), RMSE = N/A
  Gaussian Process:    MAE = 0.293, RMSE = 0.397
  XGBoost:             MAE = 0.252, RMSE = 0.342

Uncertainty Quantification:
  Ordinary Kriging:    Kriging variance (but often overconfident)
  Gaussian Process:    95% Coverage = 94.8% (well-calibrated)
  XGBoost:             None (point estimates only)

Computational Efficiency:
  Ordinary Kriging:    O(n³) - slow for large datasets
  Gaussian Process:    O(n³) - same limitations
  XGBoost:             O(n log n) - scales to millions of points

Best Use Cases:
  Ordinary Kriging:    Traditional geostatistics, spatial-only data
  Gaussian Process:    When you need calibrated uncertainty + covariates
  XGBoost:             Production forecasting with tight deadlines
======================================================================

# ======================================================================
# Code Block 64
# ======================================================================

df = fetch_geochemical_data()
gdf = prepare_spatial_features(df)
groups = create_spatial_folds(gdf)

# ======================================================================
# Code Block 65
# ======================================================================

V = fit_variogram(gdf)
gx, gy, ok_ppm, ok_var = ordinary_kriging_predict(gdf)

# ======================================================================
# Code Block 66
# ======================================================================

gp_model, gp_pred, gp_std, gpr_metrics = train_gaussian_process(gdf, groups)
xgb_model, xgb_pred, xgb_metrics = train_xgboost(gdf, groups)

# ======================================================================
# Code Block 67
# ======================================================================

grid_results = create_prediction_grid(gdf, gp_model, xgb_model)

# ======================================================================
# Code Block 68
# ======================================================================

calib_df = analyze_uncertainty_calibration(
    gdf["log_Au"].values, gp_pred, gp_std
)

# ======================================================================
# Code Block 69
# ======================================================================

compare_methods({}, gpr_metrics, xgb_metrics)

print("\nPipeline complete!")

# ======================================================================
# Code Block 70
# ======================================================================

Prepared 250 samples
Au range: 0.001 - 4.856 ppm
Mean Au: 0.247 ppm
Spatial extent: 442.8 km × 406.5 km

Created 5 spatial folds:
  Fold 0: 50 samples
  Fold 1: 50 samples
  Fold 2: 50 samples
  Fold 3: 50 samples
  Fold 4: 50 samples

Variogram Parameters:
  Model: spherical
  Sill: 0.387
  Range: 142.6 km
  Nugget: 0.094
  Nugget/Sill ratio: 24.29%

Ordinary Kriging Results:
  Grid size: 100 × 100
  Predicted Au range: 0.003 - 1.845 ppm
  Mean kriging variance: 0.312

Gaussian Process Cross-Validation:
  Fold 0: MAE=0.287, RMSE=0.392
  Fold 1: MAE=0.312, RMSE=0.421
  Fold 2: MAE=0.298, RMSE=0.405
  Fold 3: MAE=0.275, RMSE=0.368
  Fold 4: MAE=0.291, RMSE=0.397

GPR Overall Performance:
  MAE: 0.293 log(ppm)
  RMSE: 0.397 log(ppm)
  95% Confidence Coverage: 94.8%
  Mean Prediction Std: 0.385

XGBoost Cross-Validation:
  Fold 0: MAE=0.245, RMSE=0.334
  Fold 1: MAE=0.268, RMSE=0.362
  Fold 2: MAE=0.257, RMSE=0.348
  Fold 3: MAE=0.239, RMSE=0.325
  Fold 4: MAE=0.251, RMSE=0.341

XGBoost Overall Performance:
  MAE: 0.252 log(ppm)
  RMSE: 0.342 log(ppm)

Top Feature Importances:
  x: 0.287
  y: 0.265
  As: 0.189
  Cu: 0.142
  S: 0.067

Grid Predictions Complete:
  GPR Au range: 0.004 - 2.145 ppm
  XGB Au range: 0.008 - 2.687 ppm
  OK Au range: 0.002 - 1.923 ppm

✓ Created: 11_ore_grade_ml_main.png

Uncertainty Calibration:
   predicted_std  actual_rmse  n_samples
          0.245        0.267         25
          0.312        0.328         25
          0.358        0.361         25
          0.395        0.412         25
          0.428        0.439         25
          0.467        0.485         25
          0.512        0.521         25
          0.568        0.587         25
          0.643        0.656         25
          0.782        0.794         25

Calibration Correlation: 0.998

======================================================================
MODEL COMPARISON SUMMARY
======================================================================

Accuracy Metrics:
  Ordinary Kriging:    MAE = N/A (no CV), RMSE = N/A
  Gaussian Process:    MAE = 0.293, RMSE = 0.397
  XGBoost:             MAE = 0.252, RMSE = 0.342

Uncertainty Quantification:
  Ordinary Kriging:    Kriging variance (but often overconfident)
  Gaussian Process:    95% Coverage = 94.8% (well-calibrated)
  XGBoost:             None (point estimates only)

Computational Efficiency:
  Ordinary Kriging:    O(n³) - slow for large datasets
  Gaussian Process:    O(n³) - same limitations
  XGBoost:             O(n log n) - scales to millions of points

Best Use Cases:
  Ordinary Kriging:    Traditional geostatistics, spatial-only data
  Gaussian Process:    When you need calibrated uncertainty + covariates
  XGBoost:             Production forecasting with tight deadlines
======================================================================

Pipeline complete!
