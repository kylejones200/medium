# Linear Algebra for Fast Inference

This project demonstrates using linear algebra for fast inference in machine learning and data science.

## Article

Medium article: [Linear Algebra for Fast Inference](https://medium.com/@kylejones_47003/linearalgebrafastinference)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Linear algebra inference functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- System dimensions (n variables, m equations)
- Inference method (lstsq, pinv)
- Output settings

## Linear Algebra Methods

Inference methods:
- **Least Squares (lstsq)**: Standard least squares solution
- **Pseudoinverse (pinv)**: Moore-Penrose pseudoinverse
- **Fast Computation**: Optimized linear algebra operations

## Caveats

- By default, generates synthetic linear systems.
- Method selection depends on system properties.
- Large systems may require memory optimization.
