# State Space Models and Kalman Filtering for Time Series

This project demonstrates state space models and Kalman filtering for time series analysis.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Kalman filter functions
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

## Configuration

Edit `config.yaml` to customize model parameters and output settings.

## Kalman Filter

The Kalman filter:
- Estimates hidden states from noisy observations
- Recursively updates state estimates
- Provides optimal filtering under Gaussian assumptions

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette
