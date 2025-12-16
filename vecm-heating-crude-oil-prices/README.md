# Modeling Heating Oil and Crude Oil Price Relationships Using VECM

This project demonstrates Vector Error Correction Model (VECM) for analyzing cointegrated time series.

## Article

Medium article: [Modeling Heating Oil and Crude Oil Price Relationships Using VECM](https://medium.com/@kylejones_47003/modeling-heating-oil-and-crude-oil-fuel-price-relationships-using-vector-error-correction-model-d2af5214fa31)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # VECM modeling functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data symbols and date range
- VECM model parameters (lags, cointegration rank)
- Forecast horizon
- Output settings

## VECM Model

Vector Error Correction Model:
- **Cointegration**: Long-run equilibrium relationship
- **Error Correction**: Short-run adjustment mechanism
- **Forecasting**: Multi-step ahead predictions
- **Stationarity**: Requires I(1) series with cointegration

## Caveats

- Requires internet connection to fetch data from Yahoo Finance.
- Data availability depends on Yahoo Finance API.
- VECM requires cointegrated I(1) series.
