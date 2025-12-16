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

## Caveats

- Requires a CSV file with a 'Disease_Risk' column (or configure target column in config).
- SHAP computations can be slow for large datasets.
- Model uses Random Forest by default; can be extended to other models.
