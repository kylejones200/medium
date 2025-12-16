# Using Darts for Time Series Analysis in Python

This project demonstrates time series forecasting using the Darts library, including ARIMA, Exponential Smoothing, LightGBM, LSTM, NBEATS, and FFT models.

## Article

Medium article: [Using Darts for Time Series Analysis in Python](https://medium.com/@kylejones_47003/using-darts-for-time-series-analysis-in-python-dc92e08c43e5)

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
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (FRED series ID or local file path)
- Model parameters (ARIMA order, LSTM epochs, etc.)
- Which models to run
- Output settings

Note: Set `api_key` in config.yaml or use `FRED_API_KEY` environment variable.

## Caveats

- Deep learning models (LSTM, NBEATS) are disabled by default in config.yaml due to longer training times. Enable them by setting `enabled: true`.
- FRED API requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
- Data is automatically split before scaling to prevent data leakage.
