# Time Series Analysis with statsmodels in Python

This project demonstrates time series analysis using the statsmodels library in Python, including ARIMA modeling, Holt-Winters exponential smoothing, decomposition, and stationarity testing.

## Article

Medium article: [Time Series Analysis with statsmodels in Python](https://medium.com/@kylejones_47003/time-series-analysis-with-statsmodels-in-python-ea0fce203c0a)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   └── core.py        # Pure functions for analysis
├── tests/             # Unit tests
│   └── test_core.py   # Tests for core functions
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (if using synthetic data)
- Model parameters (ARIMA order, seasonal periods, etc.)
- Which analyses to run
- Output settings

## Caveats

- By default, the script generates synthetic data. To use your own data, provide a CSV file with 'date' and 'value' columns via `--data-path`.
- The ARIMA model requires sufficient data points for reliable results (recommended: 100+ observations).
- Stationarity tests assume the time series has sufficient length for meaningful statistical inference.
