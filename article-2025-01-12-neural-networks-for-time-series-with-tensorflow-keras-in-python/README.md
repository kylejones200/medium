# Neural Networks for Time Series with TensorFlow/Keras

This project demonstrates LSTM neural networks for time series forecasting.

## Article

Medium article: [Neural Networks for Time Series with TensorFlow/Keras in Python](https://medium.com/@kylejones_47003/neural-networks-for-time-series-with-tensorflow-keras-in-python-0479918268f1)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # LSTM model functions
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

Run with default settings (generates synthetic data):
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/timeseries.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Model parameters (lag, LSTM units, epochs)
- Training settings
- Output settings

## LSTM Model

Long Short-Term Memory networks:
- Capture long-term dependencies
- Handle sequential patterns
- Suitable for time series forecasting

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- LSTM training can be slow for large datasets.
- Requires sufficient data for training and validation.
- GPU recommended for faster training but not required.
