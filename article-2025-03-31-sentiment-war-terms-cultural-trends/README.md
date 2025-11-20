# Sentiment Analysis: War Terms and Cultural Trends

This project demonstrates sentiment analysis of war-related terms and cultural trends over time.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Sentiment analysis functions
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
python main.py --data-path data/text_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Positive and negative word lists
- Output settings

## Sentiment Analysis

### Sentiment Scoring
- Counts positive and negative words
- Normalizes by text length
- Provides interpretable scores

### Trend Analysis
- Tracks sentiment over time
- Identifies cultural shifts
- Visualizes patterns

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic text data.
- Simple word-counting approach; advanced NLP methods available.
- Word lists should be domain-specific for best results.
