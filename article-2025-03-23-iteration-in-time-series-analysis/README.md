# Iteration in Time Series Analysis

This project demonstrates iterative approaches to time series analysis and forecasting.

## Article

Medium article: [Iteration of Time Series Analysis and Forecasting](https://medium.com/@kylejones_47003/iteratiation-of-time-series-analysis-and-forecasting-a8eb17f37d52)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Iterative analysis functions
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
- Number of iterations
- Forecast horizon
- Output settings

## Iterative Methods

Iterative approaches:
- Refine forecasts through multiple iterations
- Incorporate feedback from previous predictions
- Improve accuracy over time

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- Iteration strategy depends on the specific problem.
- Convergence criteria should be defined for production use.
