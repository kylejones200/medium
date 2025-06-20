import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Fetch data from FRED
start_date, end_date = '2010-01-01', '2022-12-31'
df = pd.concat([
    web.DataReader('UNRATE', 'fred', start_date, end_date),
    web.DataReader('PCE', 'fred', start_date, end_date)
], axis=1).rename(columns={'UNRATE': 'unemployment_rate', 'PCE': 'consumer_spending'})

# Reset index to use date column explicitly
df = df.reset_index().rename(columns={'DATE': 'date'})

# Save to CSV
df.to_csv('unemployment_spending.csv', index=False)

# Plot time series
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title('Unemployment Rate and Consumer Spending Over Time')
ax1.set_xlabel('Year')
ax1.set_ylabel('Unemployment Rate (%)', color='red')
ax1.plot(df['date'], df['unemployment_rate'], color='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Consumer Spending (Billions)', color='blue')
ax2.plot(df['date'], df['consumer_spending'], color='blue')

plt.savefig('unemployment_consumer_spending.png')
plt.show()

# ADF Stationarity tests
for col in ['unemployment_rate', 'consumer_spending']:
    adf_result = adfuller(df[col])
    print(f'{col} ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}')

# First differencing
df['unemployment_rate_diff'] = df['unemployment_rate'].diff()
df['consumer_spending_diff'] = df['consumer_spending'].diff()

# ADF on differenced series
for col in ['unemployment_rate_diff', 'consumer_spending_diff']:
    adf_result = adfuller(df[col].dropna())
    print(f'{col} ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}')

# Granger causality tests
print('\nGranger Causality Tests:')
print('Does unemployment rate Granger-cause consumer spending?')
granger_test_ur_cs = grangercausalitytests(df[['consumer_spending_diff', 'unemployment_rate_diff']].dropna(), maxlag=4)

print('\nDoes consumer spending Granger-cause unemployment rate?')
granger_test_cs_ur = grangercausalitytests(df[['unemployment_rate_diff', 'consumer_spending_diff']].dropna(), maxlag=4)
