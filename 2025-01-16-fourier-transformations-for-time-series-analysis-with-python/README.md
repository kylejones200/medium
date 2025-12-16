# Fourier Transformations for Time Series Analysis with Python

This project demonstrates Fourier Transform analysis for time series data, including frequency domain analysis and noise filtering.

## Article

Medium article: [Fourier Transformations for Time Series Analysis](https://medium.com/@kylejones_47003/fourier-transformations-for-time-series-analysis-with-python-635747d1a35e)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Fourier transform functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Synthetic signal parameters (frequencies, amplitudes)
- Noise filtering threshold
- Which analyses to run

## Caveats

- The synthetic signal example demonstrates basic FFT on a combination of sine waves.
- The airline data example uses the sktime dataset loader.
- Noise filtering uses a simple threshold-based approach; more sophisticated methods are available.
