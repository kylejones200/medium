# Ensemble Models for Time Series Forecasting in Python

This project demonstrates ensemble forecasting using classification and regression techniques, including Random Forest models and ARIMA.

## Article

Medium article: [Ensemble Models for Time Series Forecasting](https://medium.com/@kylejones_47003/using-classification-and-regression-techniques-in-ensemble-models-for-time-series-forecasting-in-5c240a7a1b70)

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
- Data generation parameters
- Model parameters (test size, random state, ARIMA max order)
- Which analyses to run

## Caveats

- By default, the script generates synthetic random walk data. To use your own data, modify `generate_synthetic_data()` in `src/core.py` or add a data loading function.
- The ensemble approach uses a two-stage model: classification for direction, then regression with direction as a feature.
- ARIMA grid search can be slow for large parameter spaces. Adjust `arima_max_order` in config.yaml.
