import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
# Load sample dataset with missing values
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df.columns = ['Month', 'Passengers']  # Rename for clarity
# Create artificial missingness for demonstration
df.loc[10:15, 'Passengers'] = np.nan
df['Revenue'] = df['Passengers'] * 10 + np.random.normal(0, 50, size=len(df))
df.loc[5:10, 'Revenue'] = np.nan
# Visualize missingness
msno.bar(df)
plt.title("Missing Values per Column")
plt.savefig("missing_bar.png")
plt.show()
msno.matrix(df)
plt.title("Missingness Matrix")
plt.savefig("missing_matrix.png")
plt.show()
msno.heatmap(df)
plt.title("Missingness Correlation Heatmap")
plt.savefig("missing_heatmap.png")
plt.show()
# Basic stats
print("Missing values summary:\n", df.isnull().sum())
# Strategy 1: Drop rows with missing values
df_dropped = df.dropna()
print(f"After dropping, shape: {df_dropped.shape}")
# Strategy 2: Simple mean imputation
simple_imputer = SimpleImputer(strategy='mean')
df['Passengers_mean'] = simple_imputer.fit_transform(df[['Passengers']])
df['Revenue_mean'] = simple_imputer.fit_transform(df[['Revenue']])
# Strategy 3: Median imputation per group (simulate grouping by month)
df['Month_num'] = pd.to_datetime(df['Month']).dt.month
df['Revenue_median_group'] = df.groupby('Month_num')['Revenue'].transform(lambda x: x.fillna(x.median()))
# Strategy 4: Multivariate imputation
multi_imputer = IterativeImputer(random_state=0)
df[['Passengers_iter', 'Revenue_iter']] = multi_imputer.fit_transform(df[['Passengers', 'Revenue']])
# Strategy 5: Indicator for missing values
df['Revenue_missing'] = df['Revenue'].isna().astype(int)
df['Passengers_missing'] = df['Passengers'].isna().astype(int)
# Display final preview
print("\nImputed Data Preview:")
print(df[['Month', 'Passengers', 'Passengers_mean', 'Passengers_iter',
          'Revenue', 'Revenue_mean', 'Revenue_median_group', 'Revenue_iter',
          'Revenue_missing', 'Passengers_missing']].head(12))
# Save to CSV
df.to_csv("imputed_output.csv", index=False)