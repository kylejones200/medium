# Mathematical Foundations of Time Series Analysis

This project demonstrates mathematical foundations and statistical properties of time series analysis.

## Article

Medium article: [Mathematical Foundations of Time Series Analysis](https://medium.com/@kylejones_47003/mathematical-foundations-of-time-series-analysis-dc7e6e4b4622)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Mathematical analysis functions
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
- Maximum lag for autocorrelation
- Output settings

## Mathematical Properties

Fundamental properties:
- **Mean**: Central tendency
- **Variance**: Dispersion measure
- **Standard Deviation**: Square root of variance
- **Skewness**: Asymmetry measure
- **Kurtosis**: Tail heaviness
- **Autocorrelation**: Temporal dependence

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- Autocorrelation assumes stationarity.
- Statistical properties may vary with data characteristics.
