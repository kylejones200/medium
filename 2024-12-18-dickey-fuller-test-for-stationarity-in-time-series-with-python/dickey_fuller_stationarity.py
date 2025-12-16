# dickey_fuller_stationarity.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import logging

# Simulate a time series

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

np.random.seed(42)
x = np.cumsum(np.random.normal(loc=0, scale=1, size=200))
df = pd.DataFrame({'value': x})

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(df['value'])
plt.title("Simulated Time Series")
plt.savefig("simulated_timeseries.png")
plt.show()

# Calculate rolling statistics
rolling_mean = df['value'].rolling(window=12).mean()
rolling_std = df['value'].rolling(window=12).std()

plt.figure(figsize=(10, 6))
plt.plot(df['value'], label='Original')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='black')
plt.legend()
plt.title("Rolling Mean & Standard Deviation")
plt.savefig("rolling_stats.png")
plt.show()

# Apply Dickey-Fuller test
result = adfuller(df['value'])
logging.info("ADF Statistic:", result[0])
logging.info("p-value:", result[1])
for key, value in result[4].items():
    logging.info(f'Critical Value ({key}): {value}')