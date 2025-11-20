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
- Simulation parameters (n, omega, alpha)
- Model type (ARCH, GARCH, etc.)
- Forecast horizon
- Output settings

## ARCH Models

ARCH models capture volatility clustering:
- **Volatility Clustering**: High volatility periods followed by high volatility
- **Conditional Heteroskedasticity**: Variance depends on past squared errors
- **Forecasting**: Predict future volatility based on current conditions

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic returns with volatility clustering.
- ARCH models assume volatility depends only on past squared errors.
- For more complex dynamics, consider GARCH or other extensions.

