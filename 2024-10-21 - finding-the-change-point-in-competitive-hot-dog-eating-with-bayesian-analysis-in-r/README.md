# Finding the Change Point in Competitive Hot Dog Eating with Bayesian Analysis

This project demonstrates Bayesian change point detection techniques.

## Article

Medium article: [Finding the Change Point in Competitive Hot Dog Eating with Bayesian Analysis in R](https://medium.com/@kylejones_47003/finding-the-change-point-in-competitive-hot-dog-eating-with-bayesian-analysis-in-r-57b4dc95c97b)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Change point detection functions
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
python main.py --data-path data/hotdog_eating.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Change point parameters
- Detection method and window
- Output settings

## Change Point Detection

Methods demonstrated:
- **Sliding Window**: Compare means before/after potential change points
- **Bayesian Approach**: Prior on change point location, posterior inference
- **Uncertainty**: Quantify uncertainty in change point location

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic data with known change point.
- Full Bayesian implementation requires probabilistic programming.
- Window size affects detection sensitivity.
