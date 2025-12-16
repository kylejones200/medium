# Measuring Error in Time Series Forecasting with Python

This project demonstrates various error metrics for evaluating time series forecasts.

## Article

Medium article: [Measuring Error in Time Series Forecasting with Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Error measurement functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Output settings

## Error Metrics

### Mean Squared Error (MSE)
- Penalizes large errors more
- Sensitive to outliers

### Root Mean Squared Error (RMSE)
- Same units as the data
- Commonly used metric

### Mean Absolute Error (MAE)
- Less sensitive to outliers
- Easy to interpret

### Mean Absolute Percentage Error (MAPE)
- Scale-independent
- Useful for comparing across series

## Caveats

- By default, generates synthetic forecast data.
- MAPE can be problematic when actual values are near zero.
- Choose metrics appropriate for your use case.
