#!/usr/bin/env python3
"""
Orbit for Bayesian Time Series Forecasting in Python

This script demonstrates the Orbit-ML library for Bayesian time series forecasting,
including:
1. DLT (Dynamic Linear Trend) models
2. Damped trend models
3. KTR (Kernel Trend Regression) models
4. External regressors
5. Forecast evaluation metrics

Author: K.T. Jones
Date: 2025-01-31
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from orbit.models import DLT, KTR
from orbit.diagnostics.metrics import smape, rmse
from orbit.utils.dataset import load_iclaims

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_sample_data() -> pd.DataFrame:
    """
    Load sample insurance claims data from Orbit.
    
    Returns:
        DataFrame with 'date' and 'value' columns
    """
    data = load_iclaims()
    data["week"] = pd.to_datetime(data["week"])
    data = data.rename(columns={"week": "date", "claims": "value"})
    return data

# ============================================================================
# Model Training and Prediction Functions
# ============================================================================

def train_dlt_model(
    data: pd.DataFrame,
    seasonality: int = 52,
    damped: bool = False,
    regressor_col: Optional[list] = None
) -> Tuple[DLT, pd.DataFrame]:
    """
    Train a DLT (Dynamic Linear Trend) model.
    
    Args:
        data: DataFrame with 'date' and 'value' columns
        seasonality: Seasonality period (52 for weekly data)
        damped: Whether to use damped trend
        regressor_col: Optional list of regressor column names
        
    Returns:
        Tuple of (fitted model, predictions DataFrame)
    """
    model = DLT(
        response_col="value",
        date_col="date",
        seasonality=seasonality,
        damped=damped,
        regressor_col=regressor_col
    )
    
    model.fit(train_df=data)
    predictions = model.predict(df=data)
    
    return model, predictions

def train_ktr_model(
    data: pd.DataFrame,
    seasonality: int = 52,
    level_knot_prior: float = 0.5
) -> Tuple[KTR, pd.DataFrame]:
    """
    Train a KTR (Kernel Trend Regression) model.
    
    Args:
        data: DataFrame with 'date' and 'value' columns
        seasonality: Seasonality period (52 for weekly data)
        level_knot_prior: Prior for level knots
        
    Returns:
        Tuple of (fitted model, predictions DataFrame)
    """
    model = KTR(
        response_col="value",
        date_col="date",
        seasonality=seasonality,
        level_knot_prior=level_knot_prior
    )
    
    model.fit(train_df=data)
    predictions = model.predict(df=data)
    
    return model, predictions

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_forecast(
    true_values: pd.Series,
    predicted_values: pd.Series
) -> Dict[str, float]:
    """
    Calculate forecast evaluation metrics.
    
    Args:
        true_values: Actual values
        predicted_values: Predicted values
        
    Returns:
        Dictionary with SMAPE and RMSE metrics
    """
    smape_value = smape(true_values, predicted_values)
    rmse_value = rmse(true_values, predicted_values)
    
    return {
        'smape': smape_value,
        'rmse': rmse_value
    }

def log_metrics(metrics: Dict[str, float], model_name: str) -> None:
    """
    Log forecast metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    logging.info(f"{model_name} Forecast Metrics:")
    logging.info(f"  SMAPE: {metrics['smape']:.4f}")
    logging.info(f"  RMSE:  {metrics['rmse']:.4f}")

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_forecast_with_intervals(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    title: str = "Orbit Forecast",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot forecast with prediction intervals using Orbit's built-in plotting.
    
    Args:
        data: Original data DataFrame
        predictions: Predictions DataFrame
        title: Plot title
        output_path: Optional path to save figure
    """
    plot_predicted_data(
        data,
        predictions,
        date_col="date",
        actual_col="value",
        pred_col="prediction"
    )
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_forecast_custom(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    title: str = "Orbit Forecast",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot forecast with custom formatting.
    
    Args:
        data: Original data DataFrame
        predictions: Predictions DataFrame
        title: Plot title
        output_path: Optional path to save figure
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual values
    ax.plot(
        data["date"],
        data["value"],
        label="Actual",
        color="#4A90A4",
        linewidth=1.2
    )
    
    # Plot predictions
    ax.plot(
        predictions["date"],
        predictions["prediction"],
        label="Forecast",
        color="#D4A574",
        linewidth=1.2,
        linestyle="--"
    )
    
    # Plot prediction intervals if available
    if "prediction_5" in predictions.columns and "prediction_95" in predictions.columns:
        ax.fill_between(
            predictions["date"],
            predictions["prediction_5"],
            predictions["prediction_95"],
            alpha=0.2,
            color="#8B6F9E",
            label="95% Prediction Interval"
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# ============================================================================
# Demonstration Functions
# ============================================================================

def demo_basic_dlt(data: pd.DataFrame, output_dir: Path) -> None:
    """Demonstrate basic DLT model."""
    logging.info("Demo 1: Basic DLT Model")
    
    model, predictions = train_dlt_model(data, seasonality=52)
    
    logging.info("Prediction Intervals (first 5 rows):")
    logging.info(f"\n{predictions[['prediction', 'prediction_5', 'prediction_95']].head()}")
    
    # Evaluate
    metrics = evaluate_forecast(data["value"], predictions["prediction"])
    log_metrics(metrics, "Basic DLT")
    
    # Plot
    plot_forecast_custom(
        data,
        predictions,
        title="DLT Model Forecast",
        output_path=output_dir / "dlt_basic_forecast.png"
    )

def demo_damped_dlt(data: pd.DataFrame, output_dir: Path) -> None:
    """Demonstrate DLT model with damped trend."""
    logging.info("Demo 2: DLT Model with Damped Trend")
    
    model, predictions = train_dlt_model(data, seasonality=52, damped=True)
    
    # Evaluate
    metrics = evaluate_forecast(data["value"], predictions["prediction"])
    log_metrics(metrics, "Damped DLT")
    
    # Plot
    plot_forecast_custom(
        data,
        predictions,
        title="DLT Model with Damped Trend",
        output_path=output_dir / "dlt_damped_forecast.png"
    )

def demo_ktr_model(data: pd.DataFrame, output_dir: Path) -> None:
    """Demonstrate KTR (Kernel Trend Regression) model."""
    logging.info("Demo 3: KTR (Kernel Trend Regression) Model")
    
    model, predictions = train_ktr_model(
        data,
        seasonality=52,
        level_knot_prior=0.5
    )
    
    # Evaluate
    metrics = evaluate_forecast(data["value"], predictions["prediction"])
    log_metrics(metrics, "KTR")
    
    # Plot
    plot_forecast_custom(
        data,
        predictions,
        title="KTR Model Forecast",
        output_path=output_dir / "ktr_forecast.png"
    )

def demo_external_regressors(data: pd.DataFrame, output_dir: Path) -> None:
    """Demonstrate DLT model with external regressors."""
    logging.info("Demo 4: DLT Model with External Regressors")
    
    # Add simulated recession indicator
    # In practice, this would be real economic data
    data_with_regressor = data.copy()
    data_with_regressor["recession"] = [
        1 if i % 12 < 3 else 0 for i in range(len(data_with_regressor))
    ]
    
    logging.info(f"Added 'recession' regressor (simulated)")
    logging.info(f"Recession periods: {data_with_regressor['recession'].sum()}")
    
    model, predictions = train_dlt_model(
        data_with_regressor,
        seasonality=52,
        regressor_col=["recession"]
    )
    
    # Evaluate
    metrics = evaluate_forecast(
        data_with_regressor["value"],
        predictions["prediction"]
    )
    log_metrics(metrics, "DLT with Regressors")
    
    # Plot
    plot_forecast_custom(
        data_with_regressor,
        predictions,
        title="DLT Model with External Regressors",
        output_path=output_dir / "dlt_regressors_forecast.png"
    )

def compare_models(data: pd.DataFrame, output_dir: Path) -> None:
    """Compare different model configurations."""
    logging.info("Model Comparison")
    
    models = {}
    predictions_dict = {}
    
    # Basic DLT
    _, pred_basic = train_dlt_model(data, seasonality=52, damped=False)
    models["Basic DLT"] = evaluate_forecast(data["value"], pred_basic["prediction"])
    predictions_dict["Basic DLT"] = pred_basic
    
    # Damped DLT
    _, pred_damped = train_dlt_model(data, seasonality=52, damped=True)
    models["Damped DLT"] = evaluate_forecast(data["value"], pred_damped["prediction"])
    predictions_dict["Damped DLT"] = pred_damped
    
    # KTR
    _, pred_ktr = train_ktr_model(data, seasonality=52)
    models["KTR"] = evaluate_forecast(data["value"], pred_ktr["prediction"])
    predictions_dict["KTR"] = pred_ktr
    
    # Log comparison
    logging.info("Model Performance Comparison:")
    logging.info(f"{'Model':<20} {'SMAPE':<15} {'RMSE':<15}")
    for name, metrics in models.items():
        logging.info(f"{name:<20} {metrics['smape']:<15.4f} {metrics['rmse']:<15.4f}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(
        data["date"],
        data["value"],
        label="Actual",
        color="#4A90A4",
        linewidth=1.5
    )
    
    colors = ["#D4A574", "#8B6F9E", "#E8A87C"]
    for i, (name, pred) in enumerate(predictions_dict.items()):
        ax.plot(
            pred["date"],
            pred["prediction"],
            label=name,
            color=colors[i % len(colors)],
            linewidth=1.2,
            linestyle="--"
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend(loc='best')
    
    plt.savefig(output_dir / "model_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    # Create output directory
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logging.info("Orbit for Bayesian Time Series Forecasting in Python")
    
    # Load sample data
    logging.info("Loading sample insurance claims data...")
    data = load_sample_data()
    logging.info(f"Loaded {len(data)} observations")
    logging.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
    logging.info(f"First few rows:\n{data.head()}")
    
    # Run demonstrations
    try:
        demo_basic_dlt(data, output_dir)
        demo_damped_dlt(data, output_dir)
        demo_ktr_model(data, output_dir)
        demo_external_regressors(data, output_dir)
        compare_models(data, output_dir)
        
        logging.info(f"All demonstrations complete! Figures saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()

