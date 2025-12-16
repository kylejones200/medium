# PyCaret for Low-Code Time Series Forecasting in Python

This project demonstrates using PyCaret for low-code time series forecasting.

## Article

Medium article: [PyCaret for Low-Code Time Series Forecasting in Python](https://medium.com/@kylejones_47003/pycaret-for-low-code-time-series-forecasting-in-python-d3ceca00c2b5)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # PyCaret forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- PyCaret settings (train_size, compare_models, create_model)
- Output settings

## PyCaret Features

PyCaret provides:
- **Low-code interface**: Minimal code for complex workflows
- **Model comparison**: Automatically compare multiple models
- **Auto-tuning**: Automatic hyperparameter tuning
- **Multiple algorithms**: ARIMA, Prophet, Exponential Smoothing, etc.

## Caveats

- By default, generates synthetic time series data.
- PyCaret requires proper time series format (datetime index).
- Full PyCaret functionality requires additional setup steps.
