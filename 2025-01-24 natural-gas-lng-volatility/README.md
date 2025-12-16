# Natural Gas and LNG Volatility Analysis

This project analyzes volatility and correlations in natural gas and LNG prices.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Volatility analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Price columns to analyze
- Volatility window
- Correlation threshold
- Output settings

## Analysis Features

- **Volatility Calculation**: Rolling window volatility
- **Correlation Analysis**: Price relationships
- **Trend Identification**: Price movements over time
- **Risk Assessment**: Volatility patterns

## Caveats

- By default, generates synthetic price data.
- Volatility depends on window size.
- Correlations may vary over time periods.
