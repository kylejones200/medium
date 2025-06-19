import numpy as np
import matplotlib.pyplot as plt

# Set up consistent matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Ito's Lemma for log(S)
def ito_log_gbm(S0, mu, sigma, T, steps):
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), size=steps))
    W = np.insert(W, 0, 0)
    ln_S = np.log(S0) + (mu - 0.5 * sigma**2) * t + sigma * W
    S = np.exp(ln_S)
    return t, S, ln_S

# Simulate Ornstein-Uhlenbeck process
def simulate_ou(r0, mu, theta, sigma, T, steps):
    dt = T / steps
    r = np.zeros(steps + 1)
    r[0] = r0
    for i in range(steps):
        dr = theta * (mu - r[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        r[i+1] = r[i] + dr
    return np.linspace(0, T, steps + 1), r

# Generate standard normal from Brownian increment
def simulate_standard_normal_from_bm(T, steps):
    dt = T / steps
    dW = np.random.normal(0, np.sqrt(dt), size=steps)
    Z = dW / np.sqrt(dt)  # Standardized to N(0,1)
    return Z

# Steady-state distribution for OU
def steady_state_ou_pdf(x, mu, theta, sigma):
    var = sigma**2 / (2 * theta)
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mu)**2 / (2 * var))

# Simulate GBM and log-returns using Ito
t, S, ln_S = ito_log_gbm(S0=100, mu=0.05, sigma=0.2, T=1, steps=1000)

plt.figure(figsize=(10, 4))
plt.plot(t, S, label="GBM Price")
plt.title("GBM Simulation")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("ito_gbm_logprice.png")
plt.show()

# Simulate Ornstein-Uhlenbeck Process
t_ou, r = simulate_ou(r0=0.03, mu=0.05, theta=0.7, sigma=0.02, T=5, steps=1000)

plt.figure(figsize=(10, 4))
plt.plot(t_ou, r, label="OU Process (Interest Rate)")
plt.title("Ornstein-Uhlenbeck Simulation")
plt.xlabel("Time")
plt.ylabel("Rate")
plt.savefig("ou_process.png")
plt.show()

# Generate and plot standard normal
Z = simulate_standard_normal_from_bm(T=1, steps=10000)

plt.figure(figsize=(10, 4))
plt.hist(Z, bins=50, density=True, alpha=0.6, label="Simulated Standard Normals")
x = np.linspace(-4, 4, 200)
plt.plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2 / 2), label="PDF N(0,1)", linestyle='--')
plt.title("Standard Normal from Brownian Increments")
plt.xlabel("Z")
plt.ylabel("Density")
plt.savefig("standard_normal.png")
plt.show()

# Plot steady-state distribution
x_vals = np.linspace(-0.02, 0.10, 300)
pdf_vals = steady_state_ou_pdf(x_vals, mu=0.05, theta=0.7, sigma=0.02)

plt.figure(figsize=(10, 4))
plt.plot(x_vals, pdf_vals, label="Steady-State PDF")
plt.title("Steady-State Distribution of OU Process")
plt.xlabel("Rate")
plt.ylabel("Density")
plt.savefig("ou_steady_state.png")
plt.show()