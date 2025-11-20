# Natural Gas and LNG Volatility Analysis

This project analyzes volatility and correlations in natural gas and LNG prices.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Volatility analysis functions
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
python main.py --data-path data/energy_prices.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Price columns to analyze
- Volatility window
- Correlation threshold
- Output settings

## Analysis Features

- **Volatility Calculation**: Rolling window volatility
- **Correlation Analysis**: Price relationships
- **Trend Identification**: Price movements over time
- **Risk Assessment**: Volatility patterns

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic price data.
- Volatility depends on window size.
- Correlations may vary over time periods.
