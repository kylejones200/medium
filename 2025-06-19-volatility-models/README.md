# ARCH Framework for Volatility Models

This project demonstrates volatility modeling using ARCH (Autoregressive Conditional Heteroskedasticity) models.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # ARCH modeling functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Simulation parameters (n, omega, alpha)
- Model type (ARCH, GARCH, etc.)
- Forecast horizon
- Output settings

## ARCH Models

ARCH models capture volatility clustering:
- **Volatility Clustering**: High volatility periods followed by high volatility
- **Conditional Heteroskedasticity**: Variance depends on past squared errors
- **Forecasting**: Predict future volatility based on current conditions

## Caveats

- By default, generates synthetic returns with volatility clustering.
- ARCH models assume volatility depends only on past squared errors.
- For more complex dynamics, consider GARCH or other extensions.
