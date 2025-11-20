# Tidy Data: Long and Wide Format Transformations

This project demonstrates data reshaping operations including wide-to-long, long-to-wide, pivot tables, and groupby aggregations.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Data transformation functions
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

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (stores, months, weeks)
- Transformation options (wide_to_long, long_to_wide, pivot_table, groupby)
- Output settings

## Transformations

### Wide to Long (melt)
- Converts columns to rows
- Useful for time series analysis
- Preserves identifier variables

### Long to Wide (pivot)
- Converts rows to columns
- Useful for cross-tabulation
- Creates multi-level column structure

### Pivot Table
- Aggregates data during reshaping
- Supports multiple aggregation functions
- Handles duplicate index/column combinations

### Groupby Aggregation
- Groups data by specified columns
- Applies aggregation functions
- Returns summary statistics

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic sales data.
- Pivot operations require unique index/column combinations or aggregation.
- Groupby operations can be memory-intensive for large datasets.
