# Seasonal and Trend Decomposition Methods for Time Series Forecasting

This project demonstrates different decomposition methods for time series analysis, including additive, multiplicative, and robust decomposition techniques.

## Article

Medium article: [Seasonal and Trend Decomposition Methods](https://medium.com/@kylejones_47003/seasonal-and-trend-decomposition-methods-for-time-series-forecasting-c5d4564c981a)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Decomposition functions
│   └── plotting.py     # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (date range, frequency, seed)
- Decomposition period
- Which decomposition methods to run
- Output settings

## Caveats

- By default, the script generates synthetic data with known trend, seasonality, and noise.
- Additive decomposition assumes components add together.
- Multiplicative decomposition assumes components multiply together.
- Robust decomposition uses Savitzky-Golay filtering for trend estimation.
