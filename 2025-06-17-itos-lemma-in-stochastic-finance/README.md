# Ito's Lemma in Stochastic Finance

This project demonstrates Ito's Lemma and stochastic process simulations including Geometric Brownian Motion and Ornstein-Uhlenbeck processes.

## Article

Medium article: [Getting Started with Ito's Lemma for Stochastic Finance](https://medium.com/python-in-plain-english/getting-started-with-it%C3%B4s-lemma-for-stochastic-finance-6f2bd5202565)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Stochastic process functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- GBM simulation parameters (S0, mu, sigma, T, steps)
- Ornstein-Uhlenbeck parameters (r0, mu, theta, sigma)
- Standard normal generation parameters
- Steady-state distribution parameters
- Output settings

## Stochastic Processes

### Geometric Brownian Motion (GBM)
- Uses Ito's Lemma to simulate log(S)
- Common model for stock prices
- Parameters: drift (μ) and volatility (σ)

### Ornstein-Uhlenbeck Process
- Mean-reverting process
- Used for interest rates and volatility modeling
- Parameters: mean (μ), speed of reversion (θ), volatility (σ)

## Caveats

- Simulations use random number generation. Set seed in config for reproducibility.
- Step size affects accuracy of simulations.
- Steady-state distribution assumes long-term equilibrium.
