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
python main.py --data-path data/business_data.csv
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

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic business data.
- Assumes linear relationship between features and target.
- Feature selection important for model quality.
