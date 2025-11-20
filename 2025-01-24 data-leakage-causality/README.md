# Data Leakage, Lookahead Bias, and Causality

This project demonstrates data leakage detection and prevention techniques.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Data leakage analysis functions
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
- Feature engineering options
- Leakage comparison settings
- Output settings

## Data Leakage

Common leakage sources:
- **Lookahead Bias**: Using future information
- **Target Leakage**: Including target in features
- **Improper Scaling**: Scaling before train/test split
- **Data Snooping**: Using full dataset for feature engineering

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic time series data.
- Always split data before feature engineering.
- Validate models on holdout data.
