# Aeon for Time Series Clustering with Python

This project demonstrates time series classification using the aeon library with K-Neighbors Time Series Classifier.

## Article

Medium article: [Aeon for Time Series Clustering with Python](https://medium.com/@kylejones_47003/aeon-for-time-series-clustering-with-python-82229ac63282)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Classification functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
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

Run the analysis with default settings:
```bash
python main.py
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Dataset parameters (n_instances, n_timepoints, seed)
- Model parameters (test_size, n_neighbors)
- Output settings

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, the script generates synthetic data using aeon's example dataset generator.
- The train/test split uses `shuffle=False` to preserve temporal structure.
- K-Neighbors classifier with n_neighbors=1 uses the nearest neighbor for classification.
