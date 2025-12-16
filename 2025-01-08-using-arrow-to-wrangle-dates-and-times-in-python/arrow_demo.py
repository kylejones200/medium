
import arrow
import pandas as pd
import logging

# Create current UTC time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

now = arrow.utcnow()
logging.info("Now (UTC):", now)

# Shift time
logging.info("2 hours ago:", now.shift(hours=-2))
logging.info("Next week:", now.shift(weeks=1))

# Convert time zones
local = now.to('US/Central')
logging.info("US Central Time:", local)

# Format time
logging.info("Humanized:", local.humanize())
logging.info("Custom Format:", local.format('YYYY-MM-DD HH:mm:ss ZZ'))

# Parse string to arrow
parsed = arrow.get("2025-01-01T12:00:00-05:00")
logging.info("Parsed Time:", parsed)

# Arrow with Pandas
df = pd.DataFrame({'timestamp': [arrow.utcnow().shift(days=-i).datetime for i in range(5)]})
logging.info(df)

# Round to nearest hour
rounded = now.floor('hour')
logging.info("Rounded (floor hour):", rounded)

# Interval calculation
start = arrow.get("2025-01-01T08:00:00-05:00")
end = arrow.get("2025-01-01T11:30:00-05:00")
interval = end - start
logging.info("Duration in hours:", interval.total_seconds() / 3600)
