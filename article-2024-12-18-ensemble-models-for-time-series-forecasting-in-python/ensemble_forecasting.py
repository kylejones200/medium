
# ensemble_forecasting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import itertools

# Suppress ARIMA warnings
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)
data = pd.Series(np.cumsum(np.random.randn(200)))  # Random walk

df = pd.DataFrame({
    'value': data,
    'lag_1': data.shift(1),
    'lag_2': data.shift(2),
    'rate_of_change': data.diff()
}).dropna()

df['direction'] = (df['value'].shift(-1) > df['value']).astype(int)
df['next_value'] = df['value'].shift(-1)
df = df.dropna()

X = df[['value', 'lag_1', 'lag_2', 'rate_of_change']]
y_class = df['direction']
y_reg = df['next_value']

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_class_train)

X_train_reg = X_train.copy()
X_test_reg = X_test.copy()
X_train_reg['direction_pred'] = clf.predict(X_train)
X_test_reg['direction_pred'] = clf.predict(X_test)

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_reg, y_reg_train)

y_class_pred = clf.predict(X_test)
y_reg_pred = reg.predict(X_test_reg)

accuracy = accuracy_score(y_class_test, y_class_pred)
mae_rf = mean_absolute_error(y_reg_test, y_reg_pred)

print(f"Classification Accuracy: {accuracy:.2f}")
print(f"Regression MAE (RF): {mae_rf:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(df['value'], label='Original Time Series')
plt.title('Original Time Series')
plt.legend()
plt.savefig('original_time_series.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_reg_test.values, label='Actual', color='Blue')
plt.plot(y_reg_pred, label='RF Predictions', color='Red')
plt.title('RF Predictions vs Actual')
plt.legend()
plt.savefig('rf_predictions.png')
plt.show()

residuals_rf = y_reg_test.values - y_reg_pred
plt.figure(figsize=(10, 6))
plt.plot(residuals_rf, label='RF Residuals', color='Blue')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals - RF')
plt.legend()
plt.savefig('rf_residuals.png')
plt.show()

y_reg_train_series = y_reg_train.diff().dropna()
adf_test = adfuller(y_reg_train_series)
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
best_aic = np.inf
best_model = None
best_pdq = None

for param in pdq:
    try:
        model_arima = ARIMA(y_reg_train_series, order=param)
        results_arima = model_arima.fit()
        if results_arima.aic < best_aic:
            best_aic = results_arima.aic
            best_model = results_arima
            best_pdq = param
    except:
        continue

if best_model:
    print(f"Best ARIMA model: ARIMA{best_pdq}")
    forecast = best_model.get_forecast(steps=len(y_reg_test))
    y_pred_arima_diff = forecast.predicted_mean
    last_value = y_reg_train.iloc[-1]
    y_pred_arima = np.cumsum(y_pred_arima_diff) + last_value

    mae_arima = mean_absolute_error(y_reg_test, y_pred_arima)
    print(f"Regression MAE (ARIMA): {mae_arima:.2f}")

    residuals_arima = y_reg_test.values - y_pred_arima
    plt.figure(figsize=(10, 6))
    plt.plot(residuals_arima, label='ARIMA Residuals', color='Purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals - ARIMA')
    plt.legend()
    plt.savefig('arima_residuals.png')
    plt.show()
else:
    print("No valid ARIMA model found.")
