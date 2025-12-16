# Scaling Machine Learning Workflows by Moving from Python to Spark

This project demonstrates scaling machine learning workflows using Apache Spark.

## Article

Medium article: [Scaling Machine Learning Workflows by Moving from Python to Spark](https://medium.com/@kylejones_47003/scalingmlwithspark)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Spark ML functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Dataset size (samples, features)
- Spark configuration
- Output settings

## Spark ML Features

Spark capabilities:
- **Distributed Processing**: Handle large datasets
- **MLlib**: Machine learning library
- **Vector Assembler**: Feature preparation
- **Scalable Algorithms**: Logistic regression, etc.

## Caveats

- By default, generates synthetic large dataset.
- Full Spark implementation requires Spark cluster setup.
- PySpark requires Java runtime environment.
