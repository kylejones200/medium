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

## Caveats

- By default, generates synthetic time series data.
- PyTimeTK requires proper datetime indexing.
- Full functionality requires pytimetk package installation.
