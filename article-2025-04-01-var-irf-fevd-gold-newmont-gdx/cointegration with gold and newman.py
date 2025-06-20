
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import yfinance as yf
from datetime import datetime

# Download data
def get_data():
    # Get Newmont (NEM) data
    nem = yf.download('NEM', start='2020-01-01', end=datetime.now())

    # Get Gold prices (Continuous Futures)
    gold = yf.download('GC=F', start='2020-01-01', end=datetime.now())

    # Create DataFrame with closing prices
    data = pd.DataFrame({
        'NEM': nem['Close'],
        'Gold': gold['Close']
    })

    return data

# Get the data
data = get_data()

# Plot the original data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['NEM'], label='Newmont (NEM)')
plt.plot(data.index, data['Gold'], label='Gold (GC=F)')
plt.title("Newmont vs. Gold Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Function to test stationarity
def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} is not stationary. Differencing required.")
    else:
        print(f"{name} is stationary.")

# Test stationarity for each series
print("\nStationarity Tests:")
check_stationarity(data["NEM"], "Newmont")
check_stationarity(data["Gold"], "Gold")

# Calculate returns (first difference of log prices)
data_diff = np.log(data).diff().dropna()

# Plot differenced data (log returns)
plt.figure(figsize=(12, 6))
data_diff.plot(title="Daily Log Returns (NEM & Gold)")
plt.xlabel("Time")
plt.ylabel("Log Returns")
plt.grid()
plt.show()

# Granger Causality Test (e.g., does Gold cause NEM?)
print("\nGranger Causality Tests:")
print("\nTesting if Gold Granger-causes NEM:")
gc_test = grangercausalitytests(data_diff[['NEM', 'Gold']], maxlag=5)

# Fit VAR model
model = VAR(data_diff)
lag_order = model.select_order(maxlags=15)
print("\nLag Order Selection Criteria:\n", lag_order.summary())

# Fit model with optimal lags (using the AIC for example)
fitted_model = model.fit(lag_order.aic)
print("\nVAR Model Summary:")
print(fitted_model.summary())

# Plot residuals
residuals = fitted_model.resid
plt.figure(figsize=(12, 6))
residuals.plot(title="VAR Model Residuals")
plt.grid()
plt.show()

# Check residual independence with Durbin-Watson
print("\nDurbin-Watson Test Results:")
for i, col in enumerate(residuals.columns):
    dw_stat = durbin_watson(residuals[col])
    print(f"Durbin-Watson statistic for {col}: {dw_stat:.2f}")

# Forecast next 30 days
forecast = fitted_model.forecast(data_diff.values[-lag_order.aic:], steps=30)

# Convert forecast to DataFrame
forecast_index = pd.date_range(start=data.index[-1], periods=30, freq="B")
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)

# Convert from forecasted returns back to forecasted prices
# (We exponentiate the cumulative sum of returns and multiply by the last known price)
forecast_actual = np.exp(forecast_df.cumsum()) * data.iloc[-1]

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index[-100:], data["NEM"][-100:], label="NEM (Observed)")
plt.plot(data.index[-100:], data["Gold"][-100:], label="Gold (Observed)")
plt.plot(forecast_actual.index, forecast_actual["NEM"], label="NEM (Forecast)")
plt.plot(forecast_actual.index, forecast_actual["Gold"], label="Gold (Forecast)")
plt.title("VAR Model Forecast (NEM & Gold)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Print correlation
correlation = data.corr()
print("\nCorrelation between NEM and Gold:")
print(correlation)

# Calculate and print key statistics
print("\nKey Statistics:")
print(f"NEM daily volatility: {data['NEM'].pct_change().std() * 100:.2f}%")
print(f"Gold daily volatility: {data['Gold'].pct_change().std() * 100:.2f}%")
print(f"Correlation coefficient: {correlation.iloc[0,1]:.4f}")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from datetime import datetime

def get_data():
    # Explicitly specify auto_adjust=False
    nem = yf.download('NEM', start='2020-01-01', end=datetime.now(), auto_adjust=False)
    gold = yf.download('GC=F', start='2020-01-01', end=datetime.now(), auto_adjust=False)

    # If there's no data, raise an error or handle accordingly
    if nem.empty or gold.empty:
        raise ValueError("No data returned. Check the symbol or date range.")

    # Create DataFrame with the old 'Close' columns
    data = pd.DataFrame({
        'NEM': nem['Close'],
        'Gold': gold['Close']
    })

    return data

# Then call get_data() as usual
data = get_data()
print(data.head())


start_date = '2020-01-01'  # not ’2020–01–01’

nem = yf.download('NEM', start=start_date, end=datetime.now(), auto_adjust=False)
print(nem.head())
print(nem.columns)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import yfinance as yf
from datetime import datetime

def get_data():
    # Use only ASCII hyphens in your date string!
    nem = yf.download('NEM', start='2020-01-01', end=datetime.now(), auto_adjust=False)
    gold = yf.download('GC=F', start='2020-01-01', end=datetime.now(), auto_adjust=False)

    print("NEM columns:", nem.columns)
    print("Gold columns:", gold.columns)

    if nem.empty or gold.empty:
        raise ValueError("No data returned. Check the ticker or date range.")

    # Ensure 'Close' is in columns; otherwise, use 'Adj Close'
    if 'Close' not in nem.columns or 'Close' not in gold.columns:
        raise ValueError("'Close' not in columns. Check if 'Adj Close' is needed.")

    data = pd.DataFrame({
        'NEM': nem['Close'],
        'Gold': gold['Close']
    })
    return data

data = get_data()
print("Head of data:\n", data.head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import yfinance as yf
from datetime import datetime

def get_data():
    start_date = datetime.now()-pd.DateOffset(years=5)
    end_date = datetime.now()

    nem = yf.download('NEM', start=start_date, end=end_date, auto_adjust=False)
    gold = yf.download('GC=F', start=start_date, end=end_date, auto_adjust=False)

    # Basic checks
    if nem.empty or gold.empty:
        raise ValueError("One of the datasets is empty. Check symbols or date range.")

    # Ensure 'Close' column exists
    if 'Close' not in nem.columns or 'Close' not in gold.columns:
        raise ValueError("'Close' column missing. Check data structure.")

    data = pd.DataFrame({
        'NEM': nem['Close'],
        'Gold': gold['Close']
    })

    return data

# Get data
data = get_data()

# Plot raw prices
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['NEM'], label='Newmont (NEM)')
plt.plot(data.index, data['Gold'], label='Gold (GC=F)')
plt.title("Newmont vs. Gold Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Stationarity check
def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} is NOT stationary. Differencing needed.")
    else:
        print(f"{name} is stationary.")

print("\nStationarity Tests:")
check_stationarity(data['NEM'], "Newmont")
check_stationarity(data['Gold'], "Gold")

# Log returns
data_diff = np.log(data).diff().dropna()

# Plot returns
plt.figure(figsize=(12, 6))
data_diff.plot(title="Log Returns: NEM and Gold")
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.grid()
plt

start_date = datetime.now()-pd.DateOffset(years=5)
  end_date = datetime.now()

  GDX = yf.download('GDX', start=start_date, end=end_date, auto_adjust=False)
  #gold = yf.download('GC=F', start=start_date, end=end_date, auto_adjust=False)

gold

data = pd.DataFrame({
        'NEM': nem['Close'],
        'Gold': gold['Close']
    })

# Extract 'Close' prices and align by date
    GDX_close = GDX[['Close']].rename(columns={'Close': 'GDX'})
    gold_close = gold[['Close']].rename(columns={'Close': 'Gold'})

    # Join on index (dates), keeping only rows where both have data
    data = data.join(GDX_close, how='inner')

data.head()

# Plot raw prices
plt.figure(figsize=(12, 6))
plt.plot(data.index, np.log(data['GDX']), label='Newmont (NEM)')
plt.plot(data.index, np.log(data['Gold']), label='Gold (GC=F)')
plt.title("Newmont vs. Gold Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Stationarity check
def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} is NOT stationary. Differencing needed.")
    else:
        print(f"{name} is stationary.")

print("\nStationarity Tests:")
check_stationarity(data['GDX'], "Newmont")
check_stationarity(data['Gold'], "Gold")

# Log returns
data_diff = np.log(data).diff().dropna()

# Plot returns
plt.figure(figsize=(12, 6))
data_diff.plot(title="Log Returns: NEM and Gold")
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.grid()
plt

# Granger causality test
print("\nGranger Causality Test:")
grangercausalitytests(data_diff[['GDX', 'Gold']], maxlag=5, verbose=True)

# VAR model
model = VAR(data_diff)
lag_order = model.select_order(maxlags=15)
print("\nLag Order Selection:\n", lag_order.summary())

selected_lag = lag_order.aic
fitted_model = model.fit(selected_lag)
print("\nVAR Model Summary:")
print(fitted_model.summary())

# Residuals
residuals = fitted_model.resid
plt.figure(figsize=(12, 6))
residuals.plot(title="VAR Model Residuals")
plt.grid()
plt.show()

# Durbin-Watson stats
print("\nDurbin-Watson Test for Residuals:")
for col in residuals.columns:
    stat = durbin_watson(residuals[col])
    print(f"{col}: {stat:.2f}")

# Forecast next 30 days
forecast_steps = 30
forecast = fitted_model.forecast(data_diff.values[-selected_lag:], steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, freq='B')[1:]
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)

# Convert to actual prices
forecast_prices = np.exp(forecast_df.cumsum()) * data.iloc[-1]

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index[-100:], data['GDX'][-100:], label='NEM Observed')
plt.plot(data.index[-100:], data['Gold'][-100:], label='Gold Observed')
plt.plot(forecast_prices.index, forecast_prices['GDX'], label='NEM Forecast')
plt.plot(forecast_prices.index, forecast_prices['Gold'], label='Gold Forecast')
plt.title("VAR Forecast for NEM and Gold")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Correlation and stats
correlation = data.corr()
print("\nCorrelation Matrix:")
print(correlation)

print("\nKey Stats:")
print(f"NEM Volatility: {data['GDX'].pct_change().std()*100:.2f}%")
print(f"Gold Volatility: {data['Gold'].pct_change().std()*100:.2f}%")
print(f"Correlation Coefficient: {correlation.iloc[0,1]:.4f}")

# Step 1: Resample monthly and take last closing price of each month
monthly_data = data.resample('M').last()

# Step 2: Convert to log returns
monthly_log_returns = np.log(monthly_data).diff().dropna()

# Optional: print to check
print(monthly_log_returns.head())




print("Monthly Stationarity Tests:")
for col in monthly_log_returns.columns:
    check_stationarity(monthly_log_returns[col], col)

print("\nMonthly Granger Causality Tests (Gold → GDX):")
grangercausalitytests(monthly_log_returns[['GDX', 'Gold']], maxlag=3, verbose=True)

# Step 1: Select lag order
model = VAR(monthly_log_returns)
lag_order = model.select_order(maxlags=12)
print("\nMonthly Lag Order Selection:")
print(lag_order.summary())

# Step 2: Fit model
selected_lag = lag_order.aic
fitted_model = model.fit(selected_lag)

# Step 3: Summary
print(fitted_model.summary())


# Plot impulse response: how a 1-month shock to Gold affects GDX
irf = fitted_model.irf(12)  # 12 months ahead

plt.figure(figsize=(10, 6))
irf.plot(impulse='GC=F', response='GDX')
plt.suptitle("Impulse Response: Gold → GDX (monthly shocks)", fontsize=14)
plt.tight_layout()
plt.show()

# FEVD: what % of forecast error variance in GDX is explained by Gold and NEM?
fevd = fitted_model.fevd(12)  # 12 months ahead

plt.figure(figsize=(10, 6))
fevd.plot('GDX')
plt.suptitle("Variance Decomposition of GDX", fontsize=14)
plt.tight_layout()
plt.show()

irf.plot(impulse='Gold_GC=F', response='GDX_GDX')
fevd.plot('GDX_GDX')
irf = fitted_model.irf(12)

fig, ax = plt.subplots(figsize=(10, 6))
irf.plot(impulse='Gold_GC=F', response='GDX_GDX')
ax.set_title("Impulse Response: Monthly Gold Shock → GDX Response", fontsize=14)
ax.set_ylabel("Response (Log Return)")
ax.set_xlabel("Months After Shock")
plt.grid(True)
plt.tight_layout()
plt.show()



monthly_data.columns



irf = fitted_model.irf(12)

fig, ax = plt.subplots(figsize=(10, 6))
irf.plot(impulse='Gold_GC=F', response='GDX_GDX')
ax.set_title("Impulse Response: Monthly Gold Shock → GDX Response", fontsize=14)
ax.set_ylabel("Response (Log Return)")
ax.set_xlabel("Months After Shock")
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# --- Step 1: Resample to Monthly and Compute Log Returns ---

monthly_data = data.resample('ME').last()  # Use 'ME' to avoid future warning
monthly_log_returns = np.log(monthly_data).diff().dropna()

# --- Step 2: Stationarity Test on Monthly Returns ---

print("Monthly Stationarity Tests:")
def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} is NOT stationary.")
    else:
        print(f"{name} is stationary.")

for col in monthly_log_returns.columns:
    check_stationarity(monthly_log_returns[col], col)

# --- Step 3: Granger Causality Test (Gold → GDX) ---

print("\nMonthly Granger Causality Test (Gold → GDX):")
grangercausalitytests(monthly_log_returns[['GDX', 'Gold']], maxlag=3, verbose=True)

# --- Step 4: Fit VAR Model ---

model = VAR(monthly_log_returns)
lag_order = model.select_order(maxlags=12)
print("\nLag Order Selection:")
print(lag_order.summary())

selected_lag = lag_order.aic
fitted_model = model.fit(selected_lag)
print("\nVAR Model Summary:")
print(fitted_model.summary())

# --- Step 5: Impulse Response Function (IRF) ---

irf = fitted_model.irf(12)  # 12-month horizon

fig, ax = plt.subplots(figsize=(10, 6))
irf.plot(impulse='Gold_GC=F', response='GDX_GDX')
ax.set_title("Impulse Response: Monthly Gold Shock → GDX Response", fontsize=14)
ax.set_xlabel("Months After Shock")
ax.set_ylabel("Response (Log Return)")
plt.grid(False)
plt.tight_layout()
plt.show()

# --- Step 6: Forecast Error Variance Decomposition (FEVD) ---

fevd = fitted_model.fevd(12)

fig, ax = plt.subplots(figsize=(10, 6))
fevd.plot('GDX_GDX')
ax.set_title("Forecast Error Variance Decomposition of GDX", fontsize=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

fevd = fitted_model.fevd(12)


# Get the variance decomposition array (shape: [steps, variables])
gdx_index = fitted_model.names.index('GDX_GDX')
fevd_values = fevd.decomp[:, gdx_index, :]  # [time, response from each variable]

# Plot manually
plt.figure(figsize=(10, 6))
labels = fitted_model.names
for i in range(fevd_values.shape[1]):
    plt.plot(fevd_values[:, i], label=f"From {labels[i]}")

plt.title("Forecast Error Variance Decomposition: GDX")
plt.xlabel("Months Ahead")
plt.ylabel("Fraction of Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def plot_fevd_stacked(fevd, target_var, labels=None):
    import matplotlib.pyplot as plt

    idx = fitted_model.names.index(target_var)
    decomp = fevd.decomp[:, idx, :]  # shape: (time, sources)
    months = np.arange(1, decomp.shape[0] + 1)

    plt.figure(figsize=(10, 6))
    plt.stackplot(months, decomp.T, labels=labels or fitted_model.names)
    plt.title(f"FEVD - Stacked Area Chart: {target_var}")
    plt.xlabel("Months Ahead")
    plt.ylabel("Fraction of Variance")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_fevd_stacked(fevd, 'GDX_GDX', labels=['From NEM', 'From Gold', 'From GDX'])
plot_fevd_stacked(fevd, 'Gold_GC=F', labels=['From NEM', 'From Gold', 'From GDX'])
plot_fevd_stacked(fevd, 'NEM_NEM', labels=['From NEM', 'From Gold', 'From GDX'])

# Create cumulative IRFs (integrated response over time)
c_irf = fitted_model.irf(12).cumsum()

# Plot cumulative effect of a Gold shock on each variable
for target in ['NEM_NEM', 'Gold_GC=F', 'GDX_GDX']:
    fig, ax = plt.subplots(figsize=(10, 6))
    c_irf.plot(impulse='Gold_GC=F', response=target, ax=ax)
    ax.set_title(f"Cumulative IRF: Gold → {target}")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_xlabel("Months After Shock")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Compute IRF and extract orthogonalized impulse responses
irf = fitted_model.irf(12)
orth_irfs = irf.orth_irfs  # shape: (steps, variables, variables)

# Cumulative IRFs
cumulative_irfs = np.cumsum(orth_irfs, axis=0)

# Plot cumulative impact of a shock to Gold on all variables
impulse_idx = fitted_model.names.index('Gold_GC=F')
horizon = cumulative_irfs.shape[0]
months = np.arange(1, horizon + 1)

for i, response_var in enumerate(fitted_model.names):
    plt.figure(figsize=(10, 6))
    plt.plot(months, cumulative_irfs[:, i, impulse_idx], label=f"Gold → {response_var}", color='blue')
    plt.axhline(0, color='black', linewidth=0.7, linestyle='--')
    plt.title(f"Cumulative IRF: Shock to Gold → {response_var}")
    plt.xlabel("Months Ahead")
    plt.ylabel("Cumulative Response")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

