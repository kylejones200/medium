# Cross-Validation for Time Series Data

This project demonstrates various cross-validation strategies for time series data, including time series splits, rolling windows, nested CV, and blocking methods.

## Article

Medium article: [Cross-Validation for Time Series Data](https://medium.com/@kylejones_47003/cross-validation-for-time-series-data-51fd11c38e2b)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Cross-validation classes
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

Run with example data:
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/my_data.csv
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data source and column names
- Cross-validation parameters (n_splits, window_size, block_size)
- Which CV methods to run
- Analysis options

## Available CV Methods

- **TimeSeriesCV**: Standard time series cross-validation splits
- **RollingWindowCV**: Rolling window approach
- **NestedTimeSeriesCV**: Nested CV for hyperparameter tuning
- **BlockingTimeSeriesCV**: Block-based CV
- **TimeSeriesEvaluation**: Custom metrics evaluation

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- Time series CV preserves temporal ordering (no shuffling).
- Data leakage checks verify that training data comes before test data.
- Rolling window CV can be computationally expensive for large datasets.
