# Public Policy Explainability with SHAP

This project demonstrates model explainability using SHAP (SHapley Additive exPlanations) for public policy analysis.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # SHAP analysis functions
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

Run with data file:
```bash
python main.py --data-path data/public_health_data.csv
```

Run with custom configuration:
```bash
python main.py --data-path data/public_health_data.csv --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data source and target column
- Model parameters (test size, n_estimators)
- SHAP visualization options
- Output settings

## SHAP Analysis

SHAP (SHapley Additive exPlanations) provides:
- **Feature Importance**: Which features matter most
- **Feature Effects**: How each feature affects predictions
- **Individual Explanations**: Why specific predictions were made

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- Requires a CSV file with a 'Disease_Risk' column (or configure target column in config).
- SHAP computations can be slow for large datasets.
- Model uses Random Forest by default; can be extended to other models.

