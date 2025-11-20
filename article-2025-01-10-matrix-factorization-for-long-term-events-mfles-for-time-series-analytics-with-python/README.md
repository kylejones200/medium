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

Run the analysis with default settings:
```bash
python main.py
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (n_series, n_timesteps, noise level)
- SVD parameters (n_components)
- Output settings

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, the script generates synthetic multivariate time series data.
- Truncated SVD reduces dimensionality while preserving variance.
- The number of components determines the compression ratio and reconstruction quality.
