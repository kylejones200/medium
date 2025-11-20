# Nonparametric Regression with KNN for Correlated Data

This project demonstrates nonparametric regression methods including k-Nearest Neighbors (KNN) and kernel smoothing for time series data.

## Article

Medium article: [Nonparametric Regression using K-NN for Correlated Data](https://medium.com/@kylejones_47003/nonparametric-regression-using-k-nn-for-correlated-data-in-python-example-with-labor-1cb71a84479b)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Nonparametric regression functions
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

Run with default temperature data:
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/temperatures.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source URL or local file
- Feature engineering (max lag)
- Model parameters (KNN neighbors, kernel bandwidth)
- Train/test split date
- Output settings

## Methods

### k-Nearest Neighbors (KNN)
- Nonparametric regression using lag features
- Predicts based on k nearest neighbors in feature space
- Suitable for nonlinear relationships

### Kernel Smoothing
- Nonparametric smoothing using kernel regression
- Automatically selects bandwidth via cross-validation
- Captures underlying trend in noisy data

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, uses daily minimum temperature data from a public repository.
- KNN performance depends on lag feature selection and number of neighbors.
- Kernel smoothing bandwidth selection can be computationally intensive for large datasets.
