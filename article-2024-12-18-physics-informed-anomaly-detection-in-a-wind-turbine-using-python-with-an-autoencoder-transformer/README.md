# Physics-Informed Anomaly Detection in Wind Turbine

This project demonstrates physics-informed anomaly detection for wind turbine data using wavelet denoising, Isolation Forest, and tensor preparation for deep learning models.

## Article

Medium article: [Physics-Informed Anomaly Detection in Wind Turbine](https://medium.com/@kylejones_47003/physics-informed-anomaly-detection-in-a-wind-turbine-using-python-with-an-autoencoder-transformer-06eb68aeb0e8)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Anomaly detection functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files and tensor chunks
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

Run the analysis with your data:
```bash
python main.py --data-path data/wind_turbine.csv
```

Run with custom configuration:
```bash
python main.py --data-path data/wind_turbine.csv --config custom_config.yaml
```

## Data Format

The input CSV should contain the following columns (or adjust in config.yaml):
- temp
- pressure
- humidity
- altitude
- voltage
- power
- rpm
- gearbox_vibration

## Configuration

Edit `config.yaml` to customize:
- Feature names
- Preprocessing parameters (wavelet type, contamination level)
- Tensor creation parameters
- Which analyses to run

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- The script requires a CSV file with wind turbine sensor data.
- Wavelet denoising uses Daubechies 6 (db6) wavelet by default.
- Isolation Forest contamination parameter controls the expected proportion of anomalies.
- Tensor chunks are saved as .npy files for use with deep learning models.
