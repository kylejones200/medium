import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import seaborn as sns
from scipy import stats

# Generate and prepare data
np.random.seed(42)
n = 500
regimes = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
data = np.array(np.random.normal(0, np.where(regimes == 0, 1, 5)))

# Create a DataFrame with all the information
df = pd.DataFrame({
    'Data': data,
    'True_Regime': regimes,
    'Time': range(n)
})

# Fit a Markov switching model
model = MarkovRegression(data, k_regimes=2, trend='c', switching_variance=True)
result = model.fit()
print(result.summary())

print("\nTransition Matrix:")
print(result.regime_transition)

# Add predicted probabilities and regimes to DataFrame
df['Predicted_Prob_High'] = result.smoothed_marginal_probabilities[:, 1]
df['Predicted_Regime'] = np.argmax(result.smoothed_marginal_probabilities, axis=1)

# Plot 1: Original Data with Regime Highlighting
plt.figure(figsize=(12, 6))
for regime in [0, 1]:
    mask = df['True_Regime'] == regime
    plt.scatter(df[mask]['Time'], df[mask]['Data'], label=f"Regime {regime}", alpha=0.6)
plt.title("Original Data with True Regimes")
plt.legend()
plt.savefig('original_data_regimes.png')
plt.close()

# Plot 2: Predicted vs True Regimes
plt.figure(figsize=(12, 6))
plt.plot(df['True_Regime'], label='True Regime', alpha=0.6)
plt.plot(df['Predicted_Regime'], label='Predicted Regime', alpha=0.6)
plt.title("True vs Predicted Regimes")
plt.legend()
plt.savefig('true_vs_predicted_regimes.png')
plt.close()

# Plot 3: Density Plot for Each Regime
plt.figure(figsize=(12, 6))
for regime in [0, 1]:
    sns.kdeplot(data=df[df['True_Regime'] == regime]['Data'], label=f"Regime {regime}")
plt.title("Density Distribution by Regime")
plt.legend()
plt.savefig('density_distribution.png')
plt.close()

# Plot 4: Transition Probability Matrix Heatmap
plt.figure(figsize=(8, 6))
transition_matrix = result.regime_transition.reshape(2, 2)
sns.heatmap(transition_matrix, annot=True, cmap='coolwarm')
plt.title("Transition Probability Matrix")
plt.xlabel("To Regime")
plt.ylabel("From Regime")
plt.savefig('transition_matrix.png')
plt.close()

# Plot 5: Model Performance Metrics
plt.figure(figsize=(8, 6))
confusion_matrix = pd.crosstab(df['True_Regime'], df['Predicted_Regime'])
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: True vs Predicted Regimes")
plt.xlabel("Predicted Regime")
plt.ylabel("True Regime")
plt.savefig('confusion_matrix.png')
plt.close()

# Print Statistics
print("\nModel Performance Metrics:")
accuracy = (df['True_Regime'] == df['Predicted_Regime']).mean()
print(f"Prediction Accuracy: {accuracy:.2%}")

print("\nRegime Statistics:")
for regime in [0, 1]:
    regime_data = df[df['True_Regime'] == regime]['Data']
    print(f"\nRegime {regime}:")
    print(f"Mean: {regime_data.mean():.2f}")
    print(f"Std: {regime_data.std():.2f}")
    print(f"Skewness: {stats.skew(regime_data):.2f}")
    print(f"Kurtosis: {stats.kurtosis(regime_data):.2f}")

print("\nAverage Duration in Each Regime:")
for regime in [0, 1]:
    regime_runs = (df['Predicted_Regime'] == regime).astype(int).groupby(
        (df['Predicted_Regime'] != df['Predicted_Regime'].shift()).cumsum()
    ).sum()
    print(f"Regime {regime}: {regime_runs.mean():.2f} periods")

# Calculate and print transition statistics
transitions = pd.DataFrame({
    'From': df['Predicted_Regime'][:-1],
    'To': df['Predicted_Regime'][1:]
})
print("\nTransition Counts:")
print(pd.crosstab(transitions['From'], transitions['To']))
