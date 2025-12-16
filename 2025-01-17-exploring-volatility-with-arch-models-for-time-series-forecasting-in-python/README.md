# Exploring Volatility with ARCH Models for Time Series Forecasting

This project demonstrates ARCH and GARCH models for volatility forecasting.

## Article

Medium article: [Exploring Volatility with ARCH Models for Time Series Forecasting in Python](https://medium.com/@kylejones_47003/exploring-volatility-with-arch-models-for-time-series-forecasting-in-python-53966b72c1ce)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # ARCH/GARCH functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- GARCH parameters (p, q, omega, alpha, beta)
- Forecast horizon
- Output settings

## ARCH/GARCH Models

### ARCH (Autoregressive Conditional Heteroskedasticity)
- Models volatility clustering
- Volatility depends on past squared errors

### GARCH (Generalized ARCH)
- Extends ARCH with lagged volatility
- More flexible and commonly used
- Captures persistence in volatility

## Caveats

- By default, generates synthetic returns with volatility clustering.
- GARCH models assume volatility follows specific dynamics.
- Model selection (p, q) requires careful consideration.
