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

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run with default settings (generates synthetic data):
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/timeseries.csv
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

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series with injected anomalies.
- Isolation Forest requires feature engineering (lagged features).
- Statistical method assumes normal distribution.
