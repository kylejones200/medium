# Using Classification Model for Time Series Forecasting with Python

This project demonstrates using classification models for time series forecasting by converting regression to classification.

## Article

Medium article: [Using Classification Model for Time Series Forecasting with Python](https://medium.com/@kylejones_47003/using-classification-model-for-time-series-forecasting-with-python-d74a1021a5c4)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Classification forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Number of bins for classification
- Lag window size
- Output settings

## Classification Approach

Converting regression to classification:
- Bin continuous values into discrete classes
- Use classification models (Random Forest, etc.)
- Can handle non-linear patterns effectively

## Caveats

- By default, generates synthetic time series data.
- Number of bins affects granularity of predictions.
- Classification loses information compared to regression.
