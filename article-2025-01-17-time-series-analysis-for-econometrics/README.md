# Time Series Analysis for Econometrics

This project demonstrates time series analysis methods for econometric modeling.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Econometric analysis functions
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

Run with default settings:
```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize analysis parameters and output settings.

## Econometric Methods

- **ADF Test**: Tests for unit roots and stationarity
- **Cointegration**: Tests for long-run equilibrium relationships
- **VAR Model**: Vector Autoregression for multivariate time series

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette
