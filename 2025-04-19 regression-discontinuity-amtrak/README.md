# Regression Discontinuity (RD) Analysis

This project demonstrates regression discontinuity design for causal inference, analyzing the effect of a policy intervention at a specific cutoff point.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # RD analysis functions
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

Run with default settings (fetches data from FRED):
```bash
python main.py
```

Run with custom configuration:
```bash
python main.py --config custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:
- Data source (FRED series codes, date ranges)
- RD design parameters (cutoff date, bandwidth)
- Bandwidth sensitivity analysis
- Placebo test cutoff
- Outcome variables to analyze
- Output settings

## Regression Discontinuity Design

RD design exploits a discontinuity at a cutoff point to identify causal effects:
1. **Running Variable**: Distance from cutoff (e.g., months from policy date)
2. **Treatment**: Binary indicator (before/after cutoff)
3. **Outcome**: Variable of interest (e.g., unemployment rate)

### Robustness Checks
- **Bandwidth Sensitivity**: Test results across different bandwidths
- **Placebo Tests**: Test for effects at alternative cutoffs (should find no effect)
- **Covariate Continuity**: Test that pre-treatment covariates don't jump at cutoff

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette
- LOWESS smoothing with confidence bands

## Caveats

- Data is fetched from FRED. Ensure internet connection.
- RD design assumes no manipulation of the running variable around the cutoff.
- Results are sensitive to bandwidth selection.
- Placebo tests should show no significant effects at alternative cutoffs.
- Covariate continuity tests validate the identifying assumption.
