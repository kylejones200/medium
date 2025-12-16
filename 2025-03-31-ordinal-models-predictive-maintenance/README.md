# Ordinal Models for Predictive Maintenance

This project demonstrates ordinal models for predictive maintenance degradation level prediction.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Ordinal model functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Sensor columns
- Number of degradation levels
- Level labels
- Output settings

## Ordinal Models

Ordinal classification:
- Predicts ordered categories (e.g., Healthy → Degraded → Warning → Critical)
- Preserves ordering information
- Useful for degradation modeling

## Caveats

- By default, generates synthetic sensor data.
- Number of levels affects granularity.
- Ordinal models should respect ordering constraints.
