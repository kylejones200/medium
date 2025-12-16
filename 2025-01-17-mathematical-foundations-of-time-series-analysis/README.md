# Mathematical Foundations of Time Series Analysis

This project demonstrates mathematical foundations and statistical properties of time series analysis.

## Article

Medium article: [Mathematical Foundations of Time Series Analysis](https://medium.com/@kylejones_47003/mathematical-foundations-of-time-series-analysis-dc7e6e4b4622)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Mathematical analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Maximum lag for autocorrelation
- Output settings

## Mathematical Properties

Fundamental properties:
- **Mean**: Central tendency
- **Variance**: Dispersion measure
- **Standard Deviation**: Square root of variance
- **Skewness**: Asymmetry measure
- **Kurtosis**: Tail heaviness
- **Autocorrelation**: Temporal dependence

## Caveats

- By default, generates synthetic time series data.
- Autocorrelation assumes stationarity.
- Statistical properties may vary with data characteristics.
