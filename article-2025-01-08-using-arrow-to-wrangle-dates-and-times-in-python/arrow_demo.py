
import arrow
import pandas as pd

# Create current UTC time
now = arrow.utcnow()
print("Now (UTC):", now)

# Shift time
print("2 hours ago:", now.shift(hours=-2))
print("Next week:", now.shift(weeks=1))

# Convert time zones
local = now.to('US/Central')
print("US Central Time:", local)

# Format time
print("Humanized:", local.humanize())
print("Custom Format:", local.format('YYYY-MM-DD HH:mm:ss ZZ'))

# Parse string to arrow
parsed = arrow.get("2025-01-01T12:00:00-05:00")
print("Parsed Time:", parsed)

# Arrow with Pandas
df = pd.DataFrame({'timestamp': [arrow.utcnow().shift(days=-i).datetime for i in range(5)]})
print(df)

# Round to nearest hour
rounded = now.floor('hour')
print("Rounded (floor hour):", rounded)

# Interval calculation
start = arrow.get("2025-01-01T08:00:00-05:00")
end = arrow.get("2025-01-01T11:30:00-05:00")
interval = end - start
print("Duration in hours:", interval.total_seconds() / 3600)
