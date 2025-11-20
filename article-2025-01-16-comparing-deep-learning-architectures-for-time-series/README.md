# Comparing Deep Learning Architectures for Time Series

This project demonstrates comparing different deep learning architectures for time series forecasting.

## Article

Medium article: [Comparing Deep Learning Architectures for Time Series](https://medium.com/@kylejones_47003/comparing-deep-learning-architectures-for-time-series-d8c3d4c8da3e)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Deep learning comparison functions
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
- Architectures to compare (LSTM, GRU, CNN)
- Model parameters (lag, train_size)
- Output settings

## Deep Learning Architectures

### LSTM (Long Short-Term Memory)
- Handles long-term dependencies
- Good for sequential patterns
- Widely used for time series

### GRU (Gated Recurrent Unit)
- Simpler than LSTM
- Faster training
- Similar performance to LSTM

### CNN (Convolutional Neural Network)
- Captures local patterns
- Efficient for certain patterns
- Can be combined with RNNs

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- Full implementations require TensorFlow/Keras or PyTorch.
- Architecture selection depends on data characteristics.
