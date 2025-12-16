# Data Visualization

This project demonstrates various data visualization techniques using matplotlib, seaborn, and other libraries.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Visualization functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (months, regions)
- Visualization options (seaborn, weekly trends, store comparisons)
- Output settings

## Visualization Types

- **Seaborn Line Plots**: Time series with multiple categories
- **Weekly Trends**: Aggregated sales trends
- **Store Comparisons**: Side-by-side store performance

## Caveats

- By default, generates synthetic sales data.
- Interactive visualizations (Plotly, Altair, Streamlit) are optional and can be enabled in config.
