
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load or simulate data
np.random.seed(0)
dates = pd.date_range(start="2024-04-01", end="2024-10-31", freq="B")
prices = 2.5 + np.cumsum(np.random.normal(0, 0.05, len(dates)))
df = pd.DataFrame({{"date": dates, "adjClose": prices}}).set_index("date")

def bollinger_bands(df, drop: bool = True, target_col: str = 'adjClose') -> pd.DataFrame:
    if drop:
        df.dropna(inplace=True)
    df['20 Day MA'] = df[target_col].rolling(20).mean()
    df['20 Day MA_lower bound'] = df['20 Day MA'] - df[target_col].rolling(20).std() * 2
    df['20 Day MA_upper bound'] = df['20 Day MA'] + df[target_col].rolling(20).std() * 2
    return df

def bb_plot(df: pd.DataFrame, target_col: str = 'adjClose'):
    x = df.index
    plt.figure(figsize=(12, 6))
    plt.fill_between(x, df['20 Day MA_lower bound'].values, df['20 Day MA_upper bound'].values, alpha=0.3, label="Bollinger Band")
    plt.plot(x, df[target_col].values, label=target_col)
    plt.plot(x, df['20 Day MA'].values, label="20 Day MA", linestyle="--")
    plt.title("Bollinger Bands for {}".format(target_col))
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("bollinger_bands_gas_prices.png")
    plt.show()

df = bollinger_bands(df)
df.dropna(inplace=True)
bb_plot(df)
