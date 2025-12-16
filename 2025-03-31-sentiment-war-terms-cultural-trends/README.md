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

## Caveats

- By default, generates synthetic text data.
- Simple word-counting approach; advanced NLP methods available.
- Word lists should be domain-specific for best results.
