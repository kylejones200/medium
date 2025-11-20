# Portfolio Management in Quantitative Finance

This project demonstrates portfolio management concepts including efficient frontier, alpha, and beta estimation.

## Article

Medium article: [Portfolio Management in Quant Finance](https://medium.com/@kylejones_47003/portfolio-management-in-quant-finance-e7625cc3fbe9)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Portfolio management functions
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

Run with default settings:
```bash
python main.py
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Asset return generation parameters
- Efficient frontier parameters (n_portfolios)
- Alpha/beta estimation parameters
- Output settings

## Concepts

### Efficient Frontier
- Set of optimal portfolios
- Maximizes return for given risk level
- Color-coded by Sharpe ratio

### Alpha and Beta
- **Alpha (α)**: Excess return above market
- **Beta (β)**: Sensitivity to market movements
- CAPM: E[R] = Rf + β(E[Rm] - Rf) + α

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- Uses synthetic data by default. Replace with real market data for practical analysis.
- Efficient frontier assumes normal distribution of returns.
- Alpha and beta estimates are sensitive to time period and market conditions.
