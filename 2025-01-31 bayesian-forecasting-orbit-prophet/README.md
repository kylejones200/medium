# Bayesian Forecasting with Orbit-ML and Prophet

This project demonstrates Bayesian forecasting techniques using Orbit-ML and Prophet.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Forecast horizon
- Model selection (Orbit, Prophet)
- Output settings

## Bayesian Forecasting

### Orbit-ML
- Bayesian structural time series
- Probabilistic forecasts
- Automatic seasonality detection
- Uncertainty quantification

### Prophet
- Additive time series model
- Handles holidays and events
- Robust to missing data
- Fast and scalable

## Caveats

- By default, generates synthetic time series data.
- Full implementations require orbit-ml and prophet packages.
- Model performance depends on data characteristics.
