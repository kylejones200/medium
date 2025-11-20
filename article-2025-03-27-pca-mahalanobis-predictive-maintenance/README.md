# PCA and Mahalanobis Distance for Predictive Maintenance

This project demonstrates using Principal Component Analysis and Mahalanobis distance for predictive maintenance and health monitoring.

## Article

Medium article: [PCA and Mahalanobis Distance for Predictive Maintenance](https://medium.com/@kylejones_47003/principal-component-analysis-and-mahalanobis-distance-to-create-early-warnings-for-predictive-099aace0eb69)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # PCA and Mahalanobis functions
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

Run with CMAPSS data:
```bash
python main.py --data-path data/train_FD001.txt
```

Run with custom configuration:
```bash
python main.py --data-path data/train_FD001.txt --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data source and separator
- Selected sensors for PCA
- Number of PCA components
- Health index threshold
- Units to plot
- Output settings

## Methods

### Principal Component Analysis (PCA)
- Reduces dimensionality of sensor data
- Captures most variance with fewer components
- Identifies dominant patterns

### Mahalanobis Distance
- Measures distance from normal operating state
- Accounts for correlation between features
- Higher distance indicates degradation

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- Requires CMAPSS dataset or similar format.
- Threshold selection is critical for early warning.
- PCA assumes linear relationships; nonlinear methods may be needed for complex systems.
