#!/usr/bin/env python3
"""
Python code extracted from 01_time_series_forecasting_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import pandas as pd
import numpy as np

# Load the data
plants = pd.read_parquet('egrid_all_plants_1996-2023.parquet')

# Aggregate to national level
yearly_data = plants.groupby('data_year').agg({
    'Plant annual net generation (MWh)': 'sum',
    'Plant annual CO2 emissions (tons)': 'sum',
}).reset_index()

yearly_data['carbon_intensity'] = (
    yearly_data['Plant annual CO2 emissions (tons)'] / 
    yearly_data['Plant annual net generation (MWh)']
)

print(f"Data spans {yearly_data['data_year'].min()} to {yearly_data['data_year'].max()}")
print(f"Total emissions declined from {yearly_data.iloc[0]['Plant annual CO2 emissions (tons)']:,.0f} to {yearly_data.iloc[-1]['Plant annual CO2 emissions (tons)']:,.0f} tons")

# ======================================================================
# Code Block 2
# ======================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Prepare sequences (use past 3 years to predict next year)
def create_sequences(data, lookback=3):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train_data[['total_co2_tons']])

# Create sequences
X_train, y_train = create_sequences(scaled_data, lookback=3)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2)

# ======================================================================
# Code Block 3
# ======================================================================

import xgboost as xgb

def create_time_features(df):
    df = df.copy()
    
    # Lag features
    df['co2_lag1'] = df['total_co2_tons'].shift(1)
    df['co2_lag2'] = df['total_co2_tons'].shift(2)
    df['co2_lag3'] = df['total_co2_tons'].shift(3)
    
    # Rolling statistics
    df['co2_rolling_mean_3y'] = df['total_co2_tons'].rolling(3).mean()
    df['co2_rolling_std_3y'] = df['total_co2_tons'].rolling(3).std()
    
    # Trend features
    df['co2_diff1'] = df['total_co2_tons'].diff(1)
    df['co2_diff2'] = df['total_co2_tons'].diff(2)
    
    # Time-based
    df['years_since_start'] = df['year'] - df['year'].min()
    
    return df

# Create features and train
features = create_time_features(yearly_data)
feature_cols = ['co2_lag1', 'co2_lag2', 'co2_lag3', 
                'co2_rolling_mean_3y', 'co2_rolling_std_3y',
                'co2_diff1', 'years_since_start']

X_train = features[feature_cols].dropna()
y_train = features.loc[X_train.index, 'total_co2_tons']

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8
)

xgb_model.fit(X_train, y_train)

# ======================================================================
# Code Block 4
# ======================================================================

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# ======================================================================
# Code Block 5
# ======================================================================

from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# Grid search for optimal parameters
best_aic = np.inf
best_params = None

for p, d, q in itertools.product(range(3), range(2), range(3)):
    try:
        model = SARIMAX(train_data['total_co2_tons'], 
                       order=(p, d, q))
        results = model.fit(disp=False)
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = (p, d, q)
    except:
        continue

print(f"Best SARIMA order: {best_params}")

# Train final model
sarima = SARIMAX(train_data['total_co2_tons'], 
                 order=best_params)
sarima_results = sarima.fit()

# Forecast
forecast = sarima_results.forecast(steps=3)

# ======================================================================
# Code Block 6
# ======================================================================

# Average predictions from all three models
ensemble_predictions = (
    lstm_predictions.flatten() + 
    xgb_predictions + 
    sarima_predictions
) / 3

# ======================================================================
# Code Block 7
# ======================================================================

from sklearn.linear_model import LinearRegression

# Stack predictions as features
X_meta = np.column_stack([
    lstm_predictions,
    xgb_predictions,
    sarima_predictions
])

# Train meta-learner
meta_model = LinearRegression()
meta_model.fit(X_meta, test_actuals)

# Optimal weights
print("Model weights:")
for i, name in enumerate(['LSTM', 'XGBoost', 'SARIMA']):
    print(f"  {name}: {meta_model.coef_[i]:.3f}")

# ======================================================================
# Code Block 8
# ======================================================================

# Retrain on full dataset
full_model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
full_features = create_time_features(all_data)
X_full = full_features[feature_cols].dropna()
y_full = full_features.loc[X_full.index, 'total_co2_tons']
full_model.fit(X_full, y_full)

# Iteratively forecast future years
future_predictions = []
for year in range(2024, 2031):
    # Use previous predictions as features
    X_future = create_features_for_year(year, recent_history)
    prediction = full_model.predict(X_future)[0]
    future_predictions.append(prediction)
    recent_history.append(prediction)

forecast_df = pd.DataFrame({
    'Year': range(2024, 2031),
    'Predicted_CO2': future_predictions
})
print(forecast_df)

# ======================================================================
# Code Block 9
# ======================================================================

from sklearn.ensemble import GradientBoostingRegressor

# Train models for 10th, 50th, and 90th percentiles
quantile_models = {}
for quantile in [0.1, 0.5, 0.9]:
    model = GradientBoostingRegressor(
        loss='quantile', 
        alpha=quantile,
        n_estimators=100
    )
    model.fit(X_full, y_full)
    quantile_models[quantile] = model

# Generate prediction intervals
intervals = pd.DataFrame({
    'Year': range(2024, 2031),
    'Lower_10%': [quantile_models[0.1].predict(X)[0] for X in X_futures],
    'Median': [quantile_models[0.5].predict(X)[0] for X in X_futures],
    'Upper_90%': [quantile_models[0.9].predict(X)[0] for X in X_futures]
})

# ======================================================================
# Code Block 10
# ======================================================================

X, y = [], []
for i in range(lookback, len(data)):
    X.append(data[i-lookback:i, 0])
    y.append(data[i, 0])
return np.array(X), np.array(y)

# ======================================================================
# Code Block 11
# ======================================================================

LSTM(50, activation='relu', return_sequences=True, input_shape=(3, 1)),
Dropout(0.2),
LSTM(50, activation='relu'),
Dropout(0.2),
Dense(25, activation='relu'),
Dense(1)

# ======================================================================
# Code Block 12
# ======================================================================

df = df.copy()

# ======================================================================
# Code Block 13
# ======================================================================

df['co2_lag1'] = df['total_co2_tons'].shift(1)
df['co2_lag2'] = df['total_co2_tons'].shift(2)
df['co2_lag3'] = df['total_co2_tons'].shift(3)

# ======================================================================
# Code Block 14
# ======================================================================

df['co2_rolling_mean_3y'] = df['total_co2_tons'].rolling(3).mean()
df['co2_rolling_std_3y'] = df['total_co2_tons'].rolling(3).std()

# ======================================================================
# Code Block 15
# ======================================================================

df['co2_diff1'] = df['total_co2_tons'].diff(1)
df['co2_diff2'] = df['total_co2_tons'].diff(2)

# ======================================================================
# Code Block 16
# ======================================================================

df['years_since_start'] = df['year'] - df['year'].min()

return df

# ======================================================================
# Code Block 17
# ======================================================================

n_estimators=100,
max_depth=3,
learning_rate=0.1,
subsample=0.8

# ======================================================================
# Code Block 18
# ======================================================================

try:
    model = SARIMAX(train_data['total_co2_tons'], 
                   order=(p, d, q))
    results = model.fit(disp=False)
    if results.aic < best_aic:
        best_aic = results.aic
        best_params = (p, d, q)
except:
    continue

# ======================================================================
# Code Block 19
# ======================================================================

order=best_params)

# ======================================================================
# Code Block 20
# ======================================================================

print(f"  {name}: {meta_model.coef_[i]:.3f}")

# ======================================================================
# Code Block 21
# ======================================================================

X_future = create_features_for_year(year, recent_history)
prediction = full_model.predict(X_future)[0]
future_predictions.append(prediction)
recent_history.append(prediction)

# ======================================================================
# Code Block 22
# ======================================================================

model = GradientBoostingRegressor(
    loss='quantile', 
    alpha=quantile,
    n_estimators=100
)
model.fit(X_full, y_full)
quantile_models[quantile] = model
