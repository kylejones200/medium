# Martingales in Quantitative Finance

This project demonstrates martingale theory in quantitative finance, including binomial models, exponential martingales, and Girsanov's theorem.

## Article

Medium article: [Martingales in Quantitative Finance](https://medium.com/insiderfinance/martingales-in-quantitative-finance-2edba81c22a5)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Martingale simulation functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Binomial tree parameters (S0, u, d, r, steps)
- Exponential martingale parameters (theta, T, steps)
- Girsanov transformation parameters
- Output settings

## Concepts

### Binomial Model
- Risk-neutral probability measure
- Up and down movements with probabilities q and (1-q)
- Used for option pricing

### Exponential Martingale
- Z(t) = exp(-θW(t) - 0.5θ²t)
- Key tool in change of measure
- Used in Girsanov's theorem

### Girsanov's Theorem
- Changes probability measure by shifting Brownian motion
- W̃(t) = W(t) + θt
- Fundamental in risk-neutral pricing

## Caveats

- Simulations use random number generation. Set seed in config for reproducibility.
- Binomial model assumes risk-neutral probabilities.
- Girsanov transformation demonstrates measure change but doesn't optimize portfolios.
