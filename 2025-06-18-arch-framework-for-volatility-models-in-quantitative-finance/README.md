# ARCH Framework for Volatility Models in Quantitative Finance

This project demonstrates ARCH and GARCH models for volatility modeling in quantitative finance.

## Article

Medium article: [ARCH Framework for Volatility Models in Quantitative Finance](https://medium.com/@kylejones_47003/arch-framework-for-volatility-models-in-quantitative-finance-2dba4155ee09)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # ARCH/GARCH functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- GARCH parameters (p, q)
- Forecast horizon
- Output settings

## ARCH/GARCH Models

ARCH framework:
- **ARCH**: Autoregressive Conditional Heteroskedasticity
- **GARCH**: Generalized ARCH
- **Volatility clustering**: Models volatility persistence
- **Forecasting**: Predicts future volatility

## Caveats

- By default, generates synthetic returns with volatility clustering.
- GARCH models assume specific volatility dynamics.
- Model selection (p, q) requires careful consideration.
