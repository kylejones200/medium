# Gas Prices vs Unemployment Driving Analysis

This project analyzes the relationship between gas prices and unemployment rates.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Economic analysis functions
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
python main.py --data-path data/economic_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Column names
- Correlation threshold
- Output settings

## Analysis

This analysis examines:
- Correlation between gas prices and unemployment
- Linear relationship modeling
- Economic trend identification
- Policy implications

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic economic data.
- Correlation does not imply causation.
- Economic relationships vary by region and time period.
