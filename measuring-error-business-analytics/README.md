# Why Measuring Error Matters in Business Analytics (and How to Do It)

This project demonstrates error measurement techniques for business analytics.

## Article

Medium article: [Why Measuring Error Matters in Business Analytics (and How to Do It)](https://medium.com/@kylejones_47003/why-measuring-error-matters-in-business-analytics-and-how-to-do-it-2ef47d2d5dc1)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Error measurement functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Model noise level
- Output settings

## Error Metrics

Comprehensive error measurement:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Mean/Std Error**: Error distribution

## Caveats

- By default, generates synthetic prediction data.
- Error metrics should be interpreted in context.
- MAPE may be undefined for zero actual values.
