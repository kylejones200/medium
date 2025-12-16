# Causal Inference in Time Series Econometrics

This project demonstrates causal inference methods for time series econometrics.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Causal inference functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize analysis parameters and output settings.

## Causal Inference Methods

- **Granger Causality**: Tests if one time series helps predict another
- **Difference-in-Differences**: Estimates treatment effects using control groups
