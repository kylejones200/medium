# Using Darts for Time Series Analysis in Python

This project demonstrates time series forecasting using the Darts library, including ARIMA, Exponential Smoothing, LightGBM, LSTM, NBEATS, and FFT models.

## Article

Medium article: [Using Darts for Time Series Analysis in Python](https://medium.com/@kylejones_47003/using-darts-for-time-series-analysis-in-python-dc92e08c43e5)

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

Run with FRED API (requires API key):
```bash
export FRED_API_KEY=your_api_key_here
python main.py
```

Or provide API key via command line:
```bash
python main.py --api-key your_api_key_here
```

Run with local data file:
```bash
python main.py --data-path data/my_data.csv
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data source (FRED series ID or local file path)
- Model parameters (ARIMA order, LSTM epochs, etc.)
- Which models to run
- Output settings

Note: Set `api_key` in config.yaml or use `FRED_API_KEY` environment variable.

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles with key metrics
- Muted, professional color palette

## Caveats

- Deep learning models (LSTM, NBEATS) are disabled by default in config.yaml due to longer training times. Enable them by setting `enabled: true`.
- FRED API requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
- Data is automatically split before scaling to prevent data leakage.
