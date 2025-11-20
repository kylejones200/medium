# Transfer Entropy with Tech ETFs

This project demonstrates transfer entropy analysis and VECM modeling for tech ETFs.

## Article

Medium article: [What Drives the Value of Tech Stocks?](https://medium.com/@kylejones_47003/what-drives-the-value-of-tech-stocks-d86ee4f7b370)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Transfer entropy and VECM functions
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

Run with default settings (downloads data):
```bash
python main.py
```

Run with existing data:
```bash
python main.py --data-path data/xlk_smh_prices.csv
```

## Configuration

Edit `config.yaml` to customize:
- ETF tickers and date range
- ADF and Johansen test options
- VECM parameters
- Transfer entropy parameters
- Output settings

## Methods

### Vector Error Correction Model (VECM)
- Models long-run equilibrium relationships
- Captures short-run dynamics
- Error correction term shows adjustment to equilibrium

### Transfer Entropy
- Measures information flow between time series
- Directional causality measure
- Non-parametric approach

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- Requires internet connection to download data from Yahoo Finance.
- Transfer entropy computation can be slow for large datasets.
- Discretization parameters (bins) affect results.
