import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg

# --- Load and Prepare Data ---
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

# Plot the time series
data.plot(figsize=(12, 6))
plt.title('Daily Minimum Temperatures in Melbourne')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.savefig('raw_temperatures.png')
plt.show()

# --- Feature Engineering: Create lag features ---
for lag in range(1, 8):
    data[f'lag_{lag}'] = data['Temp'].shift(lag)
data.dropna(inplace=True)

# --- Split Data ---
train = data[:'1988']
test = data['1989':]
X_train = train.drop('Temp', axis=1)
y_train = train['Temp']
X_test = test.drop('Temp', axis=1)
y_test = test['Temp']

# --- Train k-NN Regressor ---
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# --- Evaluate k-NN ---
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f'k-NN Mean Squared Error: {mse_knn:.2f}')

# --- Plot k-NN results ---
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_knn, label='Predicted')
plt.title('k-NN Regression: Actual vs Predicted Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.savefig('knn_predictions.png')
plt.show()

# --- Kernel Smoothing ---
data_reset = data.reset_index()
data_reset['Time'] = np.arange(len(data_reset))  # Time variable

# Kernel regression with Gaussian kernel
kr = KernelReg(endog=[data_reset['Temp']], exog=[data_reset['Time']], var_type='c', bw='cv_ls')
y_pred_kernel, _ = kr.fit([data_reset['Time']])

# --- Plot Kernel Smoothing results ---
plt.figure(figsize=(12, 6))
plt.plot(data_reset['Date'], data_reset['Temp'], label='Original', alpha=0.5)
plt.plot(data_reset['Date'], y_pred_kernel, label='Smoothed', color='red')
plt.title('Kernel Smoothing of Daily Minimum Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.savefig('kernel_smoothing.png')
plt.show()
