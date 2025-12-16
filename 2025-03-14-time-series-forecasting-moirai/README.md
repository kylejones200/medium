# Time Series Forecasting with Moirai

This project demonstrates time series forecasting using the Moirai model from Salesforce.

## Article

Medium article: [Time Series Forecasting with Moirai](https://medium.com/@kylejones_47003/time-series-forecasting-with-moirai-5d063a40e9ed)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Moirai forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source and column names
- Model parameters (size, prediction_length, context_length)
- Test set size
- Output settings

## Moirai Model

Moirai is a foundation model for time series forecasting:
- Pre-trained on diverse time series
- Supports multiple forecast horizons
- Handles missing values and irregular frequencies
- Uses transformer architecture

## Caveats

- Requires data file with date and value columns.
- Model downloads pre-trained weights on first run (requires internet).
- GPU recommended for faster inference but not required.
- Prediction length should match test_size for proper evaluation.
