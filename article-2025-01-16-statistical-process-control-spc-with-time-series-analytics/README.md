# Statistical Process Control (SPC) with Time Series Analytics

This project demonstrates Statistical Process Control using control charts to identify out-of-control processes in time series data.

## Article

Medium article: [Statistical Process Control with Time Series Analytics](https://medium.com/@kylejones_47003/statistical-process-control-spc-with-time-series-analytics-a65b06661dc2)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # SPC functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
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

Run the analysis with default settings:
```bash
python main.py
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (date range, mean, std)
- Control limits (sigma multiplier, default 3.0)
- Output settings

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, the script generates synthetic process data with known out-of-control periods.
- Control limits are calculated using ±3σ (standard deviation) from the mean.
- Points outside control limits are flagged as out-of-control.
