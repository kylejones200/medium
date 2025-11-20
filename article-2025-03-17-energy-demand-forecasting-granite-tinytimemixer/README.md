# Energy Demand Forecasting with Granite TinyTimeMixer

This project demonstrates energy demand forecasting using IBM Granite TinyTimeMixer.

## Article

Medium article: [Forecasting Energy Demand with IBM Granite TinyTimeMixer](https://medium.com/@kylejones_47003/forecasting-energy-demand-with-ibm-granite-tinytimemixer-abd16836238a)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecasting functions
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
python main.py --data-path data/energy_demand.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Date and demand columns
- Model parameters
- Output settings

## Granite TinyTimeMixer

IBM Granite TinyTimeMixer:
- Foundation model for time series
- Pre-trained on diverse datasets
- Efficient for energy demand forecasting

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic hourly energy demand data.
- Full Granite TinyTimeMixer implementation requires additional dependencies.
- Model performance depends on data quality and temporal patterns.
