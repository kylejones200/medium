# Ensemble Models for Ordered Time Series

This project demonstrates ensemble modeling approaches for ordered time series data.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Ensemble model functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Model parameters (lag, train_size)
- Ensemble method (mean, median)
- Output settings

## Ensemble Methods

### Mean Ensemble
- Average predictions from multiple models
- Reduces variance
- Simple and effective

### Median Ensemble
- Median of predictions
- Robust to outliers
- Less sensitive to extreme predictions

## Caveats

- By default, generates synthetic time series data.
- Ensemble performance depends on model diversity.
- Ordered time series requires preserving sequence.
