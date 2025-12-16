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

## Caveats

- By default, generates synthetic data with known change point.
- Full Bayesian implementation requires probabilistic programming.
- Window size affects detection sensitivity.
