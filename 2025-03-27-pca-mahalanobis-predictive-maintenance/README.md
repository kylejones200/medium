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

## Caveats

- Requires CMAPSS dataset or similar format.
- Threshold selection is critical for early warning.
- PCA assumes linear relationships; nonlinear methods may be needed for complex systems.
