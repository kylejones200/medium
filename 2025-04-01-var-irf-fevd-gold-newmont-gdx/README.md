# VAR, IRF, and FEVD Analysis: Gold, Newmont, and GDX

This project demonstrates Vector Autoregression (VAR) modeling with Impulse Response Functions (IRF) and Forecast Error Variance Decomposition (FEVD) for analyzing dynamic relationships between financial time series.

## Article

Medium article: [Dynamic Links Between Gold, Newmont, and GDX Using VAR, IRF, and FEVD](https://medium.com/@kylejones_47003/dynamic-links-between-gold-newmont-and-gdx-using-var-irf-and-fevd-in-python-a08658d8a074)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # VAR, IRF, FEVD functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (tickers, date ranges)
- VAR model parameters (max lag, lag selection criterion)
- IRF and FEVD parameters (periods, orthogonalization)
- Granger causality tests
- Output settings

## Methods

### Vector Autoregression (VAR)
- Models multiple time series simultaneously
- Captures dynamic interdependencies
- Automatic lag selection via information criteria

### Impulse Response Functions (IRF)
- Shows response of each variable to shocks in other variables
- Cumulative IRFs show long-term effects
- Useful for understanding transmission mechanisms

### Forecast Error Variance Decomposition (FEVD)
- Decomposes forecast error variance by source
- Identifies relative importance of shocks
- Measures contribution of each variable to forecast uncertainty

## Caveats

- Data is fetched from Yahoo Finance. Ensure internet connection.
- All series must be stationary (log returns are used by default).
- VAR models require sufficient data for reliable estimation.
- Results are sensitive to lag selection and model specification.
