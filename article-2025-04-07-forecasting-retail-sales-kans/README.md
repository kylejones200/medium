# Forecasting Retail Sales with KANs

This project demonstrates forecasting retail sales using Kolmogorov-Arnold Networks (KANs).

## Article

Medium article: [Forecasting Retail Sales with Kolmogorov-Arnold Networks (KANs): Beating ARIMA with Deep Function Learning](https://medium.com/@kylejones_47003/forecasting-retail-sales-with-kolmogorov-arnold-networks-kans-beating-arima-with-deep-function-40c3f8d07fb2)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecasting functions
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
python main.py --data-path data/sales.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Date and sales columns
- Model parameters (lag, train_size)
- Output settings

## KANs (Kolmogorov-Arnold Networks)

KANs are a new type of neural network:
- Learn activation functions instead of weights
- More interpretable than traditional MLPs
- Can outperform ARIMA for time series

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic sales data.
- Full KAN implementation requires additional dependencies.
- Model performance depends on data quality and preprocessing.
