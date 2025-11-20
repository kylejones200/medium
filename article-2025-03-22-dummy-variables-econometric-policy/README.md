# Dummy Variables in Econometric Policy Analysis

This project demonstrates using dummy variables to analyze policy effects in econometric models.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Dummy variable functions
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
python main.py --data-path data/policy_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Policy date
- Effect size
- Model options (include trend)
- Output settings

## Dummy Variables

Dummy variables capture:
- **Policy interventions**: Before/after policy implementation
- **Structural breaks**: Changes in relationships
- **Seasonal effects**: Time-based patterns

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic data with policy effect.
- Policy date must be within data range.
- Trend inclusion helps control for time effects.
