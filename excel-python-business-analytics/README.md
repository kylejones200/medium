# Integrating Excel and Python for Business Analytics

This project demonstrates integrating Excel and Python for business analytics workflows.

## Article

Medium article: [Integrating Excel and Python for Business Analytics](https://medium.com/@kylejones_47003/integrating-excel-and-python-for-business-analytics-53281e2985e2)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Excel-Python integration functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Excel file path and sheet name
- Synthetic data generation
- Output settings

## Excel Integration

Features:
- **Read Excel**: Load data from .xlsx files
- **Write Excel**: Export processed data
- **Multi-sheet**: Support for multiple sheets
- **Data Analysis**: Statistical analysis of Excel data

## Caveats

- By default, generates synthetic data for demonstration.
- Requires openpyxl for Excel file handling.
- Excel file format must be .xlsx.
