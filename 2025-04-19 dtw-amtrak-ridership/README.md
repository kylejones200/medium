# Hidden Patterns in Amtrak Ridership: DTW Analysis

This project uses Dynamic Time Warping (DTW) to analyze patterns in Amtrak ridership data.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # DTW analysis functions
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

Run with default settings (uses Amtrak data if available):
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/ridership_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Station, time, and value column names
- Target station for similarity search
- Output settings

## Dynamic Time Warping (DTW)

DTW finds optimal alignment between time series:
- **Warping**: Handles time distortions
- **Distance Metric**: Measures similarity regardless of phase shifts
- **Applications**: Pattern matching, clustering, anomaly detection

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, looks for Amtrak data in data directory.
- DTW computation can be slow for large datasets.
- Similarity depends on normalization and scaling.
