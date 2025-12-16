# Building an Industrial IoT Data Pipeline with AWS

This project demonstrates building an industrial IoT data pipeline using AWS services.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # IoT pipeline functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Number of sensors
- AWS services configuration
- Output settings

## AWS IoT Pipeline

AWS services for IoT:
- **IoT Core**: Device connectivity and management
- **Kinesis**: Real-time data streaming
- **S3**: Long-term data storage
- **Lambda**: Serverless data processing
- **DynamoDB**: Time series data storage

## Caveats

- By default, generates synthetic IoT sensor data.
- Full AWS pipeline requires AWS credentials and infrastructure setup.
- Real-time processing requires proper scaling configuration.
