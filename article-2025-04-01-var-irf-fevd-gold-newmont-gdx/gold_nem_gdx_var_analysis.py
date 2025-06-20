import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# --- Load daily data from Yahoo Finance ---
tickers = ['GC=F', 'NEM', 'GDX']
start = '2018-01-01'
end = '2023-01-01'
df = yf.download(tickers, start=start, end=end)['Close'].dropna()
df.columns = ['Gold', 'NEM', 'GDX']

# --- Resample to monthly and compute log returns ---
monthly_data = df.resample('M').mean()
monthly_log_returns = np.log(monthly_data).diff().dropna()

# --- ADF Stationarity Tests ---
def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: p-value = {result[1]:.4f}")
    print(f"{name} is {'stationary' if result[1] <= 0.05 else 'NOT stationary'}.")

print("\nMonthly Stationarity Tests:")
for col in monthly_log_returns.columns:
    check_stationarity(monthly_log_returns[col], col)

# --- Granger Causality Test: Gold → GDX ---
print("\nGranger Causality Test (Gold → GDX):")
grangercausalitytests(monthly_log_returns[['GDX', 'Gold']], maxlag=3, verbose=True)

# --- Fit VAR Model ---
model = VAR(monthly_log_returns)
lag_selection = model.select_order(12)
selected_lag = lag_selection.aic
fitted_model = model.fit(selected_lag)

print("\nVAR Model Summary:")
print(fitted_model.summary())

# --- Impulse Response Functions ---
irf = fitted_model.irf(12)
irf.plot(orth=False)
plt.tight_layout()
plt.savefig('irf_plot.png')
plt.show()

# --- Forecast Error Variance Decomposition ---
fevd = fitted_model.fevd(12)
fevd.plot()
plt.tight_layout()
plt.savefig('fevd_plot.png')
plt.show()

# --- Plot cumulative IRFs for Gold shocks ---
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
variables = ['GDX', 'NEM', 'Gold']
for i, var in enumerate(variables):
    irf_plot = irf.cum_effects[:, fitted_model.names.index('Gold'), i]
    axes[i].plot(range(13), irf_plot, label=f'Gold → {var}')
    axes[i].set_title(f'Cumulative IRF: Gold → {var}')
    axes[i].legend()
plt.xlabel('Months')
plt.tight_layout()
plt.savefig('cumulative_irf_gold_shocks.png')
plt.show()

# --- Residual correlation matrix ---
print("\nResidual Correlation Matrix:")
print(fitted_model.resid.corr())