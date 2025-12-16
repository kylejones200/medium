# Treasury Spread Forecasting with Darts

This project demonstrates AutoARIMA forecasting for U.S. Treasury yield spread using the Darts library.

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
- FRED series ID (default: T10Y2Y - 10Y minus 2Y Treasury spread)
- Date range for data fetching
- Forecast horizon and model parameters
- Output settings

## Caveats

- Requires internet connection to fetch FRED data via pandas_datareader.
- The model automatically selects ARIMA parameters using AutoARIMA.
- Forecast evaluation uses the last N points of the series as hold-out data.
