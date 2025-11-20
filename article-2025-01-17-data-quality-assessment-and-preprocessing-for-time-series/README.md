# Data Quality Assessment and Preprocessing for Time Series

This project demonstrates data quality assessment and preprocessing techniques for time series data.

## Article

Medium article: [Data Quality Assessment and Preprocessing for Time Series](https://medium.com/@kylejones_47003/data-quality-assessment-and-preprocessing-for-time-series-59af0a237dc7)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Data quality functions
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

Run with default settings (generates synthetic data with issues):
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
- Preprocessing options (missing values, duplicates, outliers)
- Outlier handling method
- Output settings

## Data Quality Features

Assessment metrics:
- **Missing values**: Count and percentage
- **Duplicates**: Duplicate row detection
- **Outliers**: Statistical outlier detection
- **Data range**: Min-max spread
- **Variance**: Data variability

Preprocessing steps:
- Forward/backward fill for missing values
- Duplicate removal
- Outlier clipping (IQR method)

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic data with quality issues.
- IQR outlier method may be too aggressive for some datasets.
- Preprocessing should be tailored to specific use cases.
