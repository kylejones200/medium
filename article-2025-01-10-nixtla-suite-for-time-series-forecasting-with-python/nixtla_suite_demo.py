import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, HoltWinters, CrostonClassic as Croston,
    HistoricAverage, DynamicOptimizedTheta as DOT, SeasonalNaive
)
from utilsforecast.losses import mse, mape

# Set environment variable
os.environ['NIXTLA_ID_AS_COL'] = '1'

# Simulate ERCOT-style data
date_range = pd.date_range(start='2023-01-01', periods=240, freq='H')
load = np.sin(np.linspace(0, 12 * np.pi, 240)) * 20 + 100 + np.random.normal(0, 5, 240)
df = pd.DataFrame({'ds': date_range, 'y': load})
df["unique_id"] = "series1"

# Split the data into training and hold-out sets
hold_out_hours = 24
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

# AutoARIMA forecast
sf = StatsForecast(models=[AutoARIMA(season_length=24)], freq='h', n_jobs=-1)
sf.fit(train)
forecasts = sf.predict(h=len(hold_out))
forecasts['ds'] = hold_out['ds'].values

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='Historical Data', color='blue')
plt.plot(hold_out['ds'], hold_out['y'], label='Hold-Out Data', color='green')
plt.plot(forecasts['ds'], forecasts['AutoARIMA'], label='Forecast', color='red')
plt.title('Time Series Forecast with AutoARIMA')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AutoARIMA_Forecast.png")
plt.show()

# Metrics
actual = hold_out['y'].values
predicted = forecasts['AutoARIMA'].values
print(f"MSE: {np.mean((actual - predicted) ** 2):.2f}")
print(f"RMSE: {np.sqrt(np.mean((actual - predicted) ** 2)):.2f}")
print(f"MAE: {np.mean(np.abs(actual - predicted)):.2f}")
