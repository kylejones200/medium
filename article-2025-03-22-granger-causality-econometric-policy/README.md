# Granger Causality for Econometric Policy Analysis

This project demonstrates Granger causality testing to analyze causal relationships between economic time series variables.

## Article

Medium article: [Granger Causality for Econometric Analysis](https://medium.com/@kylejones_47003/granger-causality-for-econometric-analysis-of-public-policy-95d748643609)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Granger causality functions
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

Run with FRED data (default):
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/unemployment_spending.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source (FRED series codes)
- Date ranges
- Granger causality test parameters (maxlag)
- Test directions and hypotheses

## Granger Causality

Granger causality tests whether past values of one variable help predict another variable beyond what past values of the second variable can predict. Key steps:
1. Test for stationarity (ADF test)
2. Apply differencing if needed
3. Run Granger causality tests in both directions
4. Interpret results (p-values < 0.05 suggest Granger causality)

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- Data is fetched from FRED by default. Ensure internet connection.
- Both series must be stationary for valid Granger causality tests.
- Granger causality does not imply true causality, only predictive causality.
- Results depend on lag selection (maxlag parameter).
