# Basic Data Analysis using Pandas Library in Python

This project demonstrates basic data analysis techniques using the Pandas library.

## Article

Medium article: [Basic Data Analysis using Pandas Library in Python](https://medium.com/python-in-plain-english/basic-data-analysis-using-pandas-library-61ed815b834a)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Data analysis functions
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
- Analysis options
- Output settings

## Features

Basic Pandas operations:
- Data loading and exploration
- Summary statistics
- Data type analysis
- Memory usage
- Distribution plots

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic data for demonstration.
- Customize analysis columns in config.yaml.
- Supports CSV, Excel, and other Pandas-readable formats.
