# Dickey-Fuller Test for Stationarity in Time Series with Python

This project demonstrates the Augmented Dickey-Fuller (ADF) test for testing stationarity in time series data.

## Article

Medium article: [Dickey-Fuller Test for Stationarity](https://medium.com/@kylejones_47003/dickey-fuller-test-for-stationarity-in-time-series-with-python-4e4bf1953eed)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Stationarity testing functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
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

Run the analysis with default settings:
```bash
python main.py
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (n_samples, seed)
- Rolling window size
- Output settings

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, the script generates synthetic random walk data (non-stationary).
- The ADF test assumes the time series has sufficient length for meaningful statistical inference.
- A p-value ≤ 0.05 indicates the series is stationary (reject null hypothesis of non-stationarity).
