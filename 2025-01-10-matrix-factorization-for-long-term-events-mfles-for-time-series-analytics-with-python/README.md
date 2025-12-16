# Matrix Factorization for Long-Term Events (MFLE) for Time Series Analytics

This project demonstrates matrix factorization using Truncated SVD for time series analysis and reconstruction.

## Article

Medium article: [Matrix Factorization for Long-Term Events](https://medium.com/@kylejones_47003/matrix-factorization-for-long-term-events-mfles-for-time-series-analytics-with-python-71aba4800c91)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Matrix factorization functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (n_series, n_timesteps, noise level)
- SVD parameters (n_components)
- Output settings

## Caveats

- By default, the script generates synthetic multivariate time series data.
- Truncated SVD reduces dimensionality while preserving variance.
- The number of components determines the compression ratio and reconstruction quality.
