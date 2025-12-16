# Linear Regression for Business Analysis

This project demonstrates linear regression techniques for business analysis.

## Article

Medium article: [Linear Regression for Business Analysis](https://medium.com/@kylejones_47003/linear-regression-for-business-analysis-2407d9fe2942)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Linear regression functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Feature and target columns
- Output settings

## Linear Regression

Linear regression provides:
- **Coefficients**: Feature importance
- **R² Score**: Model fit quality
- **RMSE/MAE**: Prediction error metrics
- **Business insights**: Relationship understanding

## Caveats

- By default, generates synthetic business data.
- Assumes linear relationship between features and target.
- Feature selection important for model quality.
