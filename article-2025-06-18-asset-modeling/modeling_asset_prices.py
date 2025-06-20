import numpy as np
import matplotlib.pyplot as plt

# Set up matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False
    
})

# Simulate a Wiener process
def simulate_wiener_process(n_steps=1000, T=1.0):
    dt = T / n_steps
    increments = np.random.normal(0, np.sqrt(dt), size=n_steps)
    W = np.cumsum(increments)
    W = np.insert(W, 0, 0)  # Start at zero
    return W

# Simulate a geometric Brownian motion
def simulate_gbm(S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=1000):
    dt = T / n_steps
    W = simulate_wiener_process(n_steps, T)
    t = np.linspace(0, T, n_steps + 1)
    exponent = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(exponent)
    return t, S

# Plot the Wiener process
W = simulate_wiener_process()
t_W = np.linspace(0, 1, len(W))

plt.figure(figsize=(10, 4))
plt.plot(t_W, W, label="Wiener Process")
plt.xlabel("Time")
plt.ylabel("W(t)")
plt.title("Simulated Wiener Process")
plt.grid(False)
plt.savefig("wiener_process.png")
plt.show()

# Plot the GBM
t, S = simulate_gbm()

plt.figure(figsize=(10, 4))
plt.plot(t, S, label="Geometric Brownian Motion")
plt.xlabel("Time")
plt.ylabel("S(t)")
plt.title("Simulated Geometric Brownian Motion")
plt.grid(False)
plt.savefig("gbm_simulation.png")
plt.show()