# Using Arrow to Wrangle Dates and Times in Python

This project demonstrates date and time manipulation using the Arrow library.

## Article

Medium article: [Using Arrow to Wrangle Dates and Times in Python](https://medium.com/@kylejones_47003/using-arrow-to-wrangle-dates-and-times-in-python-05f2e08de508)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Arrow date/time functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Time operations (shifts, timezone conversions)
- Format strings
- Interval calculations
- Output settings

## Arrow Features

Arrow provides intuitive date/time operations:
- **Time Shifting**: Add/subtract time periods
- **Timezone Conversion**: Convert between timezones
- **Humanization**: Natural language time descriptions
- **Parsing**: Parse various date/time formats
- **Rounding**: Round to specified precision
- **Intervals**: Calculate time differences

## Caveats

- Arrow is a drop-in replacement for datetime with better API.
- Timezone conversions require valid timezone names.
- Humanization is relative to current time.
