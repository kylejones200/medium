# Alternative Metrics for Predictive Maintenance

This project demonstrates alternative metrics for predictive maintenance beyond traditional RUL estimation.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Alternative metrics functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Sensor columns
- Health threshold
- Degradation window
- Output settings

## Alternative Metrics

### Health Index
- Aggregated sensor readings
- Single metric for system health
- Easy to interpret

### Degradation Rate
- Rate of health decline
- Early warning indicator
- Trend-based metric

### Remaining Useful Life (RUL)
- Time until failure threshold
- Based on health index
- Actionable metric

## Caveats

- By default, generates synthetic sensor data.
- Health threshold selection is critical.
- Degradation rate depends on window size.
