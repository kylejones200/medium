# Regime Switching Models for Time Series

This project demonstrates Markov switching regression models for identifying regime changes in time series data.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Regime switching functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (n_samples, regime probabilities, standard deviations)
- Model parameters (number of regimes, switching variance)
- Output settings

## Caveats

- By default, the script generates synthetic data with two regimes.
- The model uses Markov switching regression with switching variance.
- Transition probabilities are estimated from the data.
