# Time Series Visualization for Business Analysis with Python

This project demonstrates time series visualization techniques for business analysis.

## Article

Medium article: [Time Series Visualization for Business Analysis with Python](https://medium.com/@kylejones_47003/time-series-visualization-for-business-analysis-with-python-5df695543d4a)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Visualization functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Visualization options (time_series, multiple_series, seasonal_decomposition)
- Seasonal period for decomposition
- Output settings

## Visualization Types

### Time Series Plot
- Basic line plot of time series
- Clean, minimalist design

### Multiple Series
- Compare multiple time series
- Useful for business metrics

### Seasonal Decomposition
- Trend, seasonal, and residual components
- Helps understand patterns

## Caveats

- By default, generates synthetic business data.
- Seasonal period should match data frequency.
- Visualizations are optimized for business reporting.
