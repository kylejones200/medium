# Asset Price Modeling

This project demonstrates asset price modeling using Wiener processes and Geometric Brownian Motion.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Asset modeling functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Wiener process parameters (n_steps, T)
- GBM parameters (S0, mu, sigma, T, n_steps)
- Output settings

## Models

### Wiener Process (Brownian Motion)
- Continuous-time stochastic process
- Independent increments
- Normal distribution

### Geometric Brownian Motion (GBM)
- Standard model for stock prices
- S(t) = S₀ exp((μ - 0.5σ²)t + σW(t))
- Parameters: drift (μ) and volatility (σ)

## Caveats

- Simulations use random number generation. Set seed in config for reproducibility.
- GBM assumes constant drift and volatility.
- Step size affects simulation accuracy.
