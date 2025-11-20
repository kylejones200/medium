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
python main.py --data-path data/sensor_data.csv
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

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic sensor data.
- Number of levels affects granularity.
- Ordinal models should respect ordering constraints.
