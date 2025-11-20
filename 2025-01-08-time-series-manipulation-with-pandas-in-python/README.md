# Time Series Manipulation with Pandas in Python

This project demonstrates time series manipulation techniques using Pandas.

## Article

Medium article: [Time Series Manipulation with Pandas in Python](https://medium.com/@kylejones_47003/time-series-manipulation-with-pandas-in-python-ac8ffc64b670)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Time series manipulation functions
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
- Date and value column names
- Rolling window size
- Resampling frequency
- Output settings

## Features

Pandas time series operations:
- Rolling statistics (mean, std)
- Shifting and lagging
- Percentage change
- Resampling to different frequencies
- Time-based indexing

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- Requires datetime index for resampling.
- Window size affects rolling statistics.
