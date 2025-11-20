# PyTimeTK for Time Series Analysis in Python

This project demonstrates using PyTimeTK for time series analysis and manipulation.

## Article

Medium article: [PyTimeTK for Time Series Analysis in Python](https://medium.com/@kylejones_47003/pytimetk-for-time-series-analysis-in-python-92f725352d99)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # PyTimeTK analysis functions
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
- PyTimeTK features to compute
- Window size for analysis
- Output settings

## PyTimeTK Features

PyTimeTK provides:
- **Time-based grouping**: `summarize_by_time()`
- **Padding**: `pad_by_time()`
- **Rolling operations**: Time-aware rolling functions
- **Time series manipulation**: Date/time operations

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- PyTimeTK requires proper datetime indexing.
- Full functionality requires pytimetk package installation.
