# TSFresh for Time Series Feature Extraction

This project demonstrates automated feature extraction from time series data using TSFresh.

## Article

Medium article: [TSFresh for Time Series Feature Extraction in Python](https://medium.com/towardsdev/tsfresh-for-time-series-feature-extraction-in-python-c4ee791f467c)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # TSFresh feature extraction functions
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
- Data generation parameters (n_series, n_timesteps)
- Feature extraction settings (comprehensive features, FDR level)
- Feature selection options
- Output settings

## TSFresh Features

TSFresh automatically extracts hundreds of features:
- **Statistical Features**: Mean, variance, skewness, kurtosis
- **Temporal Features**: Autocorrelation, partial autocorrelation
- **Frequency Features**: FFT coefficients, spectral entropy
- **Complexity Features**: Approximate entropy, sample entropy

## Feature Selection

- Uses statistical tests to identify relevant features
- Controls False Discovery Rate (FDR)
- Reduces dimensionality while preserving information

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- Feature extraction can be computationally intensive for large datasets.
- Comprehensive feature set includes 700+ features by default.
- Feature selection requires target variable (y) for supervised learning.
