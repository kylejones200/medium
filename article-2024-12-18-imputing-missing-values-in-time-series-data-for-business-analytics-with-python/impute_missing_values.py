# impute_missing_values.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate a time series with missing values
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.cumsum(np.random.randn(100))
data[::10] = np.nan  # introduce missing values every 10 days
df = pd.DataFrame({'date': dates, 'value': data}).set_index('date')

# Visualize original data with missing values
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['value'], label='Original')
plt.title("Time Series with Missing Values")
plt.savefig('missing_values.png')
plt.show()

# Fill forward
df_ffill = df.fillna(method='ffill')

# Linear interpolation
df_interp = df.interpolate()

# Compare visually
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['value'], label='Original', alpha=0.5)
plt.plot(df.index, df_ffill['value'], label='Forward Fill', linestyle='--')
plt.plot(df.index, df_interp['value'], label='Interpolated', linestyle='-.')
plt.title("Comparison of Imputation Methods")
plt.legend()
plt.savefig('imputation_comparison.png')
plt.show()

# Summary
print("Original missing count:", df['value'].isna().sum())
print("After ffill missing count:", df_ffill['value'].isna().sum())
print("After interpolation missing count:", df_interp['value'].isna().sum())