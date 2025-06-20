
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter

# Create synthetic data with known components
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')

# Generate individual components
t = np.linspace(0, len(dates)-1, len(dates))
trend = 0.1 * t + 10
seasonality = 5 * np.sin(2 * np.pi * t / 365)
noise = np.random.normal(0, 1, len(dates))

# Combine components
data = trend + seasonality + noise
df = pd.DataFrame({'date': dates, 'value': data})
df.set_index('date', inplace=True)

# Additive vs. Multiplicative Decomposition
def compare_decomposition_methods(df, period=365):
    additive = seasonal_decompose(df['value'], period=period, model='additive')
    multiplicative = seasonal_decompose(df['value'], period=period, model='multiplicative')
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].plot(df['value']); axes[0, 0].set_title('Original (Additive)')
    axes[0, 1].plot(additive.trend); axes[0, 1].set_title('Trend')
    axes[0, 2].plot(additive.seasonal); axes[0, 2].set_title('Seasonal')
    axes[0, 3].plot(additive.resid); axes[0, 3].set_title('Residual')
    axes[1, 0].plot(df['value']); axes[1, 0].set_title('Original (Multiplicative)')
    axes[1, 1].plot(multiplicative.trend); axes[1, 1].set_title('Trend')
    axes[1, 2].plot(multiplicative.seasonal); axes[1, 2].set_title('Seasonal')
    axes[1, 3].plot(multiplicative.resid); axes[1, 3].set_title('Residual')
    plt.tight_layout()
    plt.savefig("Compare_Decomposition_Methods.png")
    plt.show()
    return additive, multiplicative

# Robust decomposition
def robust_decomposition(df, period=365):
    trend = pd.Series(savgol_filter(df['value'], window_length=period // 2 + 1, polyorder=3), index=df.index)
    detrended = df['value'] - trend
    seasonal = pd.Series(0.0, index=df.index)
    for i in range(period):
        seasonal[i::period] = detrended[i::period].median()
    residuals = df['value'] - trend - seasonal
    return pd.DataFrame({'original': df['value'], 'trend': trend, 'seasonal': seasonal, 'residual': residuals})

# Analyze components
def analyze_components(decomposition_result):
    trend = decomposition_result.trend.dropna()
    trend_direction = 'increasing' if trend.iloc[-1] > trend.iloc[0] else 'decreasing'
    trend_strength = abs(trend.iloc[-1] - trend.iloc[0]) / len(trend)
    seasonal = decomposition_result.seasonal.dropna()
    seasonal_amplitude = seasonal.max() - seasonal.min()
    residuals = decomposition_result.resid.dropna()
    residual_variance = residuals.var()
    return {
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'seasonal_amplitude': seasonal_amplitude,
        'residual_variance': residual_variance
    }

# Run decompositions and analysis
additive_decomp, mult_decomp = compare_decomposition_methods(df)
robust_results = robust_decomposition(df)
analysis_results = analyze_components(additive_decomp)
print(analysis_results)
