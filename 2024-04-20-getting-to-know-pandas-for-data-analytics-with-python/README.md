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

## Caveats

- By default, generates synthetic data for demonstration.
- Operations depend on data structure.
- Customize operations list in config.yaml.
