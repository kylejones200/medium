import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Simulate process data ---
np.random.seed(42)
time = pd.date_range(start="2023-01-01", periods=100, freq="D")
values = np.random.normal(50, 2, 100)
values[30:35] += 8  # Out-of-control high
values[70:75] -= 8  # Out-of-control low
df = pd.DataFrame({"Time": time, "Value": values})

# --- Calculate control limits ---
mean = df["Value"].mean()
std_dev = df["Value"].std()
ucl = mean + 3 * std_dev
lcl = mean - 3 * std_dev
out_of_control = (df["Value"] > ucl) | (df["Value"] < lcl)

# --- Plot control chart ---
plt.figure(figsize=(12, 6))
plt.plot(df["Time"], df["Value"], label="Process Data", marker="o", linestyle="-")
plt.axhline(mean, color="blue", linestyle="--", label="Mean")
plt.axhline(ucl, color="red", linestyle="--", label="Upper Control Limit (UCL)")
plt.axhline(lcl, color="red", linestyle="--", label="Lower Control Limit (LCL)")
plt.scatter(df["Time"][out_of_control], df["Value"][out_of_control], color="red", label="Out of Control")
plt.fill_between(df["Time"], ucl, lcl, where=out_of_control, color="red", alpha=0.1)
plt.title("Control Chart with Out-of-Control Areas")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("control_chart_outliers.png")
plt.show()
