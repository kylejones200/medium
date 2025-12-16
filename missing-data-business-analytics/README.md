# Working with Missing Data in Business Analytics

This project demonstrates techniques for handling missing data in business analytics.

## Article

Medium article: [Working with Missing Data in Business Analytics](https://medium.com/@kylejones_47003/working-with-missing-data-in-business-analytics-74b1f5f0efd2)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Missing data analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (URL, file path, or synthetic generation)
- Imputation strategy (mean, median, iterative)
- Output settings

## Missing Data Strategies

Imputation methods:
- **Mean/Median**: Simple univariate imputation
- **Iterative**: Multivariate imputation using other features
- **Group-based**: Imputation within groups
- **Indicator variables**: Track missingness patterns

## Caveats

- By default, generates synthetic data with artificial missingness.
- Imputation strategy should match data characteristics.
- Missingness patterns may indicate data quality issues.
