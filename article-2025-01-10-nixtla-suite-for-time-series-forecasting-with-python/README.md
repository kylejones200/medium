# Nixtla Suite for Time Series Forecasting with Python

This project demonstrates time series forecasting using the Nixtla StatsForecast library with AutoARIMA.

## Article

Medium article: [Nixtla Suite for Time Series Forecasting](https://medium.com/@kylejones_47003/nixtla-suite-for-time-series-forecasting-with-python-b0f318365e9b)

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
- Data generation parameters (start date, periods, frequency)
- Model parameters (season length, frequency)
- Forecast horizon
- Output settings

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles with key metrics
- Muted, professional color palette

## Caveats

- By default, the script generates synthetic ERCOT-style hourly data.
- The model uses AutoARIMA with seasonal length of 24 hours.
- StatsForecast requires the `NIXTLA_ID_AS_COL` environment variable to be set.
