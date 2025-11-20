# Remote Sensing and Public Data Earth Observation

This project demonstrates remote sensing analysis using public earth observation data.

## Article

Medium article: [How Remote Sensing and Public Data are Powering a New Era of Earth Observation](https://medium.com/@kylejones_47003/how-remote-sensing-and-public-data-are-powering-a-new-era-of-earth-observation-8002f2b1e04d)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Remote sensing functions
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

Run with default settings (generates synthetic satellite data):
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/satellite_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Number of spectral bands
- NDVI calculation options
- Output settings

## Remote Sensing Features

Analysis includes:
- **Multi-spectral bands**: Red, Green, Blue, NIR, SWIR
- **NDVI calculation**: Vegetation index
- **Time series analysis**: Temporal patterns
- **Public data integration**: Earth observation datasets

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic satellite data.
- Real remote sensing data available from USGS, NASA, ESA.
- NDVI requires Red and NIR bands.
