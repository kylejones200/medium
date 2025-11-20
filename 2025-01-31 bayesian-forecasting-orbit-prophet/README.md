# Bayesian Forecasting with Orbit-ML and Prophet

This project demonstrates Bayesian forecasting techniques using Orbit-ML and Prophet.

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
- Forecast horizon
- Model selection (Orbit, Prophet)
- Output settings

## Bayesian Forecasting

### Orbit-ML
- Bayesian structural time series
- Probabilistic forecasts
- Automatic seasonality detection
- Uncertainty quantification

### Prophet
- Additive time series model
- Handles holidays and events
- Robust to missing data
- Fast and scalable

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- Full implementations require orbit-ml and prophet packages.
- Model performance depends on data characteristics.
