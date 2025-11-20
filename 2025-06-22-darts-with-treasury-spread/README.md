# Treasury Spread Forecasting with Darts

This project demonstrates AutoARIMA forecasting for U.S. Treasury yield spread using the Darts library.

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

Run the analysis with default settings (T10Y2Y series):
```bash
python main.py
```

Run with custom series and forecast horizon:
```bash
python main.py --series-id T10Y2Y --forecast-horizon 30
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- FRED series ID (default: T10Y2Y - 10Y minus 2Y Treasury spread)
- Date range for data fetching
- Forecast horizon and model parameters
- Output settings

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles with key metrics
- Muted, professional color palette

## Caveats

- Requires internet connection to fetch FRED data via pandas_datareader.
- The model automatically selects ARIMA parameters using AutoARIMA.
- Forecast evaluation uses the last N points of the series as hold-out data.

