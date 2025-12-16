# Copula Methods for Multivariate Time Series

This project demonstrates copula methods for modeling dependencies in multivariate time series.

## Article

Medium article: [Copula Methods for Modeling Dependency in Multivariate Time Series](https://medium.com/@kylejones_47003/copula-methods-for-modeling-dependency-in-multivariate-time-series-in-python-with-examples-from-360ebf3d202b)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Copula analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (time_steps, seed)
- Copula types (Clayton, Student-t)
- Forecast sample sizes
- Output settings

## Copula Methods

### Clayton Copula
- Asymmetric dependence structure
- Lower tail dependence
- Useful for modeling joint extremes

### Student-t Copula
- Symmetric dependence
- Tail dependence in both directions
- More flexible than Gaussian copula

## Caveats

- By default, generates synthetic data for demonstration.
- Copula fitting requires sufficient data for reliable parameter estimation.
- Forecasts preserve dependence structure but may not capture all dynamics.
