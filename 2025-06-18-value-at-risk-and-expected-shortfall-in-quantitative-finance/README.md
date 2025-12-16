# Value at Risk and Expected Shortfall in Quantitative Finance

This project demonstrates Value at Risk (VaR) and Expected Shortfall (ES) calculations for risk management.

## Article

Medium article: [Value at Risk (VaR) and Expected Shortfall in Quantitative Finance](https://medium.com/@kylejones_47003/value-at-risk-var-and-expected-shortfall-in-quantitative-finance-76fab1b35b35)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # VaR and ES functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- VaR methods (parametric, historical, monte_carlo)
- Confidence levels
- Output settings

## Risk Metrics

### Value at Risk (VaR)
- **Parametric**: Assumes normal distribution
- **Historical**: Uses actual historical returns
- **Monte Carlo**: Simulates from return distribution

### Expected Shortfall (ES)
- Also known as Conditional VaR (CVaR)
- Average loss beyond VaR threshold
- More conservative than VaR

## Caveats

- By default, generates synthetic returns for demonstration.
- Parametric VaR assumes normal distribution; may underestimate tail risk.
- Historical VaR requires sufficient historical data.
- Monte Carlo VaR depends on distribution assumptions.
