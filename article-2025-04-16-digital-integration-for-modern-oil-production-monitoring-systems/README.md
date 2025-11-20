# Digital Integration for Modern Oil Production Monitoring Systems

This project demonstrates digital integration systems for modern oil production monitoring.

## Article

Medium article: [Digital Integration for Modern Oil Production Monitoring Systems](https://medium.com/@kylejones_47003/digital-integration-for-modern-oil-production-monitoring-systems-b4877961aa38)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Production monitoring functions
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
python main.py --data-path data/production_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Monitoring thresholds and alerts
- Output settings

## Features

Digital integration features:
- Real-time production monitoring
- KPI calculation (efficiency, volatility)
- Alert thresholds
- Historical data analysis

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic production data.
- Production monitoring requires real-time data integration.
- Alert thresholds should be calibrated to operational requirements.
