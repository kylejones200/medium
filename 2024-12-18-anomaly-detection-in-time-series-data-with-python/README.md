# Anomaly Detection in Time Series Data with Python

This project demonstrates anomaly detection techniques for time series data.

## Article

Medium article: [Anomaly Detection in Time Series Data with Python](https://medium.com/gitconnected/anomaly-detection-in-time-series-data-with-python-5a15089636db)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Anomaly detection functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Detection method (isolation_forest, statistical)
- Model parameters (contamination, threshold)
- Output settings

## Anomaly Detection Methods

### Isolation Forest
- Unsupervised learning approach
- Handles multivariate data
- Effective for high-dimensional spaces

### Statistical Method (Z-score)
- Simple and interpretable
- Based on standard deviations
- Fast computation

## Caveats

- By default, generates synthetic time series with injected anomalies.
- Isolation Forest requires feature engineering (lagged features).
- Statistical method assumes normal distribution.
