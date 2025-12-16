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

## Caveats

- By default, generates synthetic time series data.
- Iteration strategy depends on the specific problem.
- Convergence criteria should be defined for production use.
