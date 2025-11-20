# Survival Analysis Time to Failure

This project demonstrates survival analysis for modeling time-to-failure in predictive maintenance.

## Article

Medium article: [Predictive Maintenance: Modeling Time to Failure Using Survival Analysis in Python](https://medium.com/@kylejones_47003/predictive-maintenance-modeling-time-to-failure-using-survival-analysis-in-python-turbofan-35dac4415bac)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Survival analysis functions
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
python main.py --data-path data/survival_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Duration and event columns
- Model selection (Kaplan-Meier, Weibull, Cox PH)
- Output settings

## Survival Analysis Methods

### Kaplan-Meier Estimator
- Non-parametric survival curve
- Handles censored data
- No distributional assumptions

### Weibull Model
- Parametric survival model
- Flexible hazard function
- Provides distribution parameters

### Cox Proportional Hazards
- Semi-parametric model
- Incorporates covariates
- Estimates hazard ratios

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic survival data.
- Requires duration and event columns in data.
- Censoring is important for accurate survival estimates.
