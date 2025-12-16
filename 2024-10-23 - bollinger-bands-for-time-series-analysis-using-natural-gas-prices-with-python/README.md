# Bollinger Bands for Time Series Analysis

This project demonstrates Bollinger Bands analysis for time series data, commonly used in technical analysis of financial markets.

## Article

Medium article: [Bollinger Bands for Time Series Analysis](https://medium.com/datadriveninvestor/bollinger-bands-for-time-series-analysis-using-natural-gas-prices-with-python-f0d13181b26f)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Bollinger Bands functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (or synthetic generation parameters)
- Bollinger Bands parameters (window size, number of standard deviations)
- Output settings

## Bollinger Bands

Bollinger Bands consist of:
- **Middle Band**: N-period moving average
- **Upper Band**: Middle band + (N-period standard deviation × multiplier)
- **Lower Band**: Middle band - (N-period standard deviation × multiplier)

Prices touching the upper or lower bands may indicate overbought or oversold conditions.

## Caveats

- By default, the script generates synthetic price data if no data file is provided.
- Bollinger Bands are most effective with sufficient historical data (at least 20 periods for default window).
- The bands adapt to volatility, expanding during volatile periods and contracting during stable periods.
