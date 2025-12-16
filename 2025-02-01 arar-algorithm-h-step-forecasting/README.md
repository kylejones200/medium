# H-Step Forecasting with the ARAR Algorithm

This project demonstrates the ARAR (AutoRegressive AutoRegressive) algorithm for h-step ahead forecasting, with comparison to ARIMA.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # ARAR forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source and column names
- ARAR model parameters (lag selection strategy, lags)
- ARIMA model parameters (order)
- Forecast horizon
- Output settings

## ARAR Algorithm

The ARAR algorithm:
1. Applies differencing to remove trend
2. Selects reduced lag set (typically powers of 2: 1, 2, 4, 8, 16)
3. Fits autoregressive model on differenced data
4. Generates h-step forecasts
5. Reverses differencing to reconstruct original scale

## Caveats

- The algorithm requires sufficient data for differencing and lag selection.
- Reduced lag sets (powers of 2) are computationally efficient but may not capture all dependencies.
- ARAR is compared with ARIMA to demonstrate relative performance.
