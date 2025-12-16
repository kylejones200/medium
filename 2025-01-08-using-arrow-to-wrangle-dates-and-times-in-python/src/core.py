"""Core functions for date and time wrangling with Arrow."""

import arrow
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_current_time() -> arrow.Arrow:
    """Get current UTC time."""
    return arrow.utcnow()

def shift_time(time: arrow.Arrow, **kwargs) -> arrow.Arrow:
    """Shift time by specified amount."""
    return time.shift(**kwargs)

def convert_timezone(time: arrow.Arrow, timezone: str) -> arrow.Arrow:
    """Convert time to specified timezone."""
    return time.to(timezone)

def format_time(time: arrow.Arrow, format_str: str = 'YYYY-MM-DD HH:mm:ss ZZ') -> str:
    """Format time using custom format string."""
    return time.format(format_str)

def humanize_time(time: arrow.Arrow) -> str:
    """Get humanized time string."""
    return time.humanize()

def parse_time_string(time_str: str) -> arrow.Arrow:
    """Parse time string to Arrow object."""
    return arrow.get(time_str)

def round_time(time: arrow.Arrow, precision: str = 'hour') -> arrow.Arrow:
    """Round time to specified precision."""
    return time.floor(precision)

def calculate_interval(start: arrow.Arrow, end: arrow.Arrow) -> float:
    """Calculate time interval in hours."""
    return (end - start).total_seconds() / 3600

def create_time_series_dataframe(times: List[arrow.Arrow]) -> pd.DataFrame:
    """Create pandas DataFrame from Arrow time objects."""
    return pd.DataFrame({'timestamp': [t.datetime for t in times]})

def demonstrate_arrow_operations() -> Dict[str, any]:
    """Demonstrate various Arrow operations."""
    now = get_current_time()
    
    results = {
        'current_utc': now,
        'two_hours_ago': shift_time(now, hours=-2),
        'next_week': shift_time(now, weeks=1),
        'us_central': convert_timezone(now, 'US/Central'),
        'humanized': humanize_time(now),
        'formatted': format_time(now),
        'parsed': parse_time_string("2025-01-01T12:00:00-05:00"),
        'rounded': round_time(now, 'hour'),
    }
    
    start = parse_time_string("2025-01-01T08:00:00-05:00")
    end = parse_time_string("2025-01-01T11:30:00-05:00")
    results['interval_hours'] = calculate_interval(start, end)
    
    return results

