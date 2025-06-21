
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Simulate or load data
dates = pd.date_range(start="2024-04-01", end="2024-10-31", freq="B")
prices = 2.5 + np.cumsum(np.random.normal(0, 0.05, len(dates)))
df = pd.DataFrame({{"adjClose": prices}}, index=dates)

# Calculate Bollinger Bands
df['20 MA'] = df['adjClose'].rolling(20).mean()
std = df['adjClose'].rolling(20).std()
df['Lower'] = df['20 MA'] - 2 * std
df['Upper'] = df['20 MA'] + 2 * std
df.dropna(inplace=True)

# Plot
plt.figure(figsize=(12, 6))
plt.fill_between(df.index, df['Lower'], df['Upper'], alpha=0.3, label="Band")
plt.plot(df.index, df['adjClose'], label="adjClose")
plt.plot(df.index, df['20 MA'], linestyle="--", label="20 MA")
plt.title("Bollinger Bands")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.savefig("bollinger_bands_simplified.png")
plt.show()
