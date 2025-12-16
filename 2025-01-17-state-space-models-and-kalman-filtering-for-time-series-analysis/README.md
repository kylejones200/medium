# State Space Models and Kalman Filtering for Time Series

This project demonstrates state space models and Kalman filtering for time series analysis.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Kalman filter functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize model parameters and output settings.

## Kalman Filter

The Kalman filter:
- Estimates hidden states from noisy observations
- Recursively updates state estimates
- Provides optimal filtering under Gaussian assumptions
