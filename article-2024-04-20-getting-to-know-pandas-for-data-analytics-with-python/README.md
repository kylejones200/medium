# Getting to Know Pandas for Data Analytics with Python

This project demonstrates getting started with Pandas for data analytics.

## Article

Medium article: [Getting to Know Pandas for Data Analytics with Python](https://medium.com/@kylejones_47003/getting-to-know-pandas-for-data-analytics-with-python-7386da28dd33)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Pandas analytics functions
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
python main.py --data-path data/dataset.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Operations to perform (groupby, sort, filter)
- Output settings

## Pandas Operations

Common operations demonstrated:
- **GroupBy**: Aggregate data by categories
- **Sort**: Sort by values
- **Filter**: Filter based on conditions
- **Data Analysis**: Info, head, tail, missing values

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic data for demonstration.
- Operations depend on data structure.
- Customize operations list in config.yaml.
