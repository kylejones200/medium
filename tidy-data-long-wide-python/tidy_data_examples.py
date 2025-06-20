import pandas as pd
import matplotlib.pyplot as plt

# Example 1: Melt wide to long
df = pd.DataFrame({
    'Store': ['A', 'B'],
    'Jan_Sales': [100, 90],
    'Feb_Sales': [120, 100],
    'Mar_Sales': [130, 110]
})

long_df = pd.melt(df, id_vars='Store', var_name='Month', value_name='Sales')
long_df['Month'] = long_df['Month'].str.replace('_Sales', '')

# Example 2: Pivot long to wide
wide_df = long_df.pivot(index='Store', columns='Month', values='Sales').reset_index()

# Example 3: Pivot table with aggregation
pivot_df = long_df.pivot_table(
    index='Store',
    columns='Month',
    values='Sales',
    aggfunc='sum'
).reset_index()

# Example 4: Groupby sum and mean
data = pd.DataFrame({
    'Store': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Month': ['Jan', 'Feb', 'Mar', 'Jan', 'Feb', 'Mar'],
    'Sales': [100, 120, 130, 90, 100, 110]
})

store_sum = data.groupby('Store')['Sales'].sum().reset_index()
month_avg = data.groupby('Month')['Sales'].mean().reset_index()
store_month_sum = data.groupby(['Store', 'Month'])['Sales'].sum().reset_index()

# Example 5: Multiple aggregations
store_agg = data.groupby('Store')['Sales'].agg(['mean', 'sum', 'std']).reset_index()
store_renamed = data.groupby('Store')['Sales'].agg(
    avg_sales='mean',
    total_sales='sum',
    volatility='std'
).reset_index()

# Example 6: Resetting index
summary = data.groupby('Store')['Sales'].sum().reset_index()

# Practical Use Case: Weekly Sales
raw = pd.DataFrame({
    'Store': ['North', 'South', 'East', 'West'],
    'Week_1': [300, 250, 400, 375],
    'Week_2': [310, 245, 390, 380],
    'Week_3': [305, 260, 395, 370]
})

# Step 1: Reshape from wide to long
long = pd.melt(
    raw,
    id_vars='Store',
    var_name='Week',
    value_name='Sales'
)
long['Week'] = long['Week'].str.replace('Week_', '').astype(int)

# Step 2: Summarize store performance
summary = long.groupby('Store')['Sales'].agg(
    avg='mean',
    total='sum'
).reset_index()

# Step 3: Company-wide weekly sales trend
weekly = long.groupby('Week')['Sales'].sum().reset_index()

plt.plot(weekly['Week'], weekly['Sales'], marker='o')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.title('Company Sales Trend by Week')
plt.savefig('weekly_sales.png')
plt.show()

# Step 4: Compare stores side-by-side
pivot = long.pivot(index='Week', columns='Store', values='Sales').reset_index()

for store in pivot.columns[1:]:
    plt.plot(pivot['Week'], pivot[store], label=store)

plt.xlabel('Week')
plt.ylabel('Sales')
plt.title('Store-wise Weekly Sales')
plt.legend()
plt.savefig('store_weekly_sales.png')
plt.show()
