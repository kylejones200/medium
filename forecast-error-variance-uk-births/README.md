# Forecast Error Variance Analysis for Time Series Using UK Births Data

This project demonstrates forecast error variance analysis for time series forecasting.

## Article

Medium article: [Forecast Error Variance Analysis for Time Series Using UK Births Data](https://medium.com/@kylejones_47003/forecast-error-variance-analysis-for-time-series-using-historical-data-of-births-in-the-uk-from-a527646a134c)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecast error variance functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (URL, file path, or synthetic generation)
- Model parameters (seasonal periods, alpha)
- Multi-step variance coefficients
- Output settings

## Forecast Error Variance

Analysis includes:
- **Exponential Smoothing**: Fitted model
- **Error Metrics**: MAD, variance, sigma approximation
- **Multi-step Variance**: Forecast error variance growth
- **Smoothed Errors**: Moving average and exponential smoothing

## Caveats

- By default, generates synthetic UK births data.
- Requires Excel file format for original data.
- Multi-step variance coefficients depend on model assumptions.
