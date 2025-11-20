#!/usr/bin/env python3
"""
Ensemble Models for Time Series Forecasting

Main entry point for running ensemble forecasting models.

Usage:
    python main.py
    python main.py --config custom_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from src.core import (
    generate_synthetic_data,
    create_features,
    split_data,
    fit_classifier,
    fit_regressor,
    add_direction_feature,
    test_stationarity,
    find_best_arima,
    forecast_arima,
    plot_time_series,
    plot_predictions,
    plot_residuals
)
from sklearn.metrics import accuracy_score, mean_absolute_error


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Ensemble Models for Time Series Forecasting')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    data = generate_synthetic_data(config['data']['n_samples'], config['data']['seed'])
    df = create_features(data)
    
    X = df[['value', 'lag_1', 'lag_2', 'rate_of_change']]
    y_class = df['direction']
    y_reg = df['next_value']
    
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = split_data(
        X, y_class, y_reg,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )
    
    plot_time_series(df['value'], output_dir / 'original_time_series.png', 
                    "Original Time Series")
    
    if config['analysis']['run_classification']:
        print("Fitting classification model...")
        clf = fit_classifier(X_train, y_class_train, config['model']['random_state'])
        y_class_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_class_test, y_class_pred)
        print(f"Classification Accuracy: {accuracy:.2%}")
    
    if config['analysis']['run_regression']:
        print("Fitting regression model...")
        X_train_reg = add_direction_feature(X_train, clf)
        X_test_reg = add_direction_feature(X_test, clf)
        
        reg = fit_regressor(X_train_reg, y_reg_train, config['model']['random_state'])
        y_reg_pred = reg.predict(X_test_reg)
        mae_rf = mean_absolute_error(y_reg_test, y_reg_pred)
        print(f"Regression MAE (RF): {mae_rf:.2f}")
        
        plot_predictions(y_reg_test, y_reg_pred, output_dir / 'rf_predictions.png',
                        "Random Forest Predictions", {'mae': mae_rf})
        
        residuals_rf = y_reg_test.values - y_reg_pred
        plot_residuals(residuals_rf, output_dir / 'rf_residuals.png',
                      "Random Forest Residuals")
    
    if config['analysis']['run_arima']:
        print("Finding best ARIMA model...")
        y_reg_train_series = y_reg_train.diff().dropna()
        stationarity = test_stationarity(y_reg_train_series)
        print(f"ADF Statistic: {stationarity['adf_statistic']:.4f}")
        print(f"p-value: {stationarity['p_value']:.4f}")
        
        best_arima = find_best_arima(y_reg_train_series, config['model']['arima_max_order'])
        
        if best_arima:
            print(f"Best ARIMA model: ARIMA{best_arima['order']}")
            y_pred_arima = forecast_arima(
                best_arima['model'],
                len(y_reg_test),
                y_reg_train.iloc[-1]
            )
            mae_arima = mean_absolute_error(y_reg_test, y_pred_arima)
            print(f"Regression MAE (ARIMA): {mae_arima:.2f}")
            
            plot_predictions(y_reg_test, y_pred_arima, output_dir / 'arima_predictions.png',
                            f"ARIMA{best_arima['order']} Predictions", {'mae': mae_arima})
            
            residuals_arima = y_reg_test.values - y_pred_arima
            plot_residuals(residuals_arima, output_dir / 'arima_residuals.png',
                          f"ARIMA{best_arima['order']} Residuals")
        else:
            print("No valid ARIMA model found.")
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

