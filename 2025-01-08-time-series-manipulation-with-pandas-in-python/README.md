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

## Caveats

- By default, generates synthetic time series data.
- Requires datetime index for resampling.
- Window size affects rolling statistics.
