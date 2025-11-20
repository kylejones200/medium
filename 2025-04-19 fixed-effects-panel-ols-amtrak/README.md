# Fixed Effects Time Series Modeling with Panel OLS

This project demonstrates fixed effects panel OLS modeling for time series analysis.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Panel OLS functions
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

Run with default settings (uses Amtrak data if available):
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/panel_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Entity and time column names
- Fixed effects options
- Output settings

## Fixed Effects Panel OLS

Fixed effects models:
- **Entity Fixed Effects**: Control for unobserved entity-specific characteristics
- **Time Fixed Effects**: Control for common time trends
- **Panel Structure**: Multiple entities observed over time

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- Requires panel data structure (entities × time).
- Fixed effects absorb variation within entities/time.
- Clustered standard errors recommended for panel data.
