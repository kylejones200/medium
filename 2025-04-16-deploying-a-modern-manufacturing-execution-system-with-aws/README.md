# Deploying a Modern Manufacturing Execution System with AWS

This project demonstrates deploying a manufacturing execution system (MES) using AWS services.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Manufacturing system functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- AWS region and services
- Output settings

## AWS Services

Common AWS services for MES:
- **S3**: Data storage
- **Lambda**: Serverless compute
- **DynamoDB**: NoSQL database
- **IoT Core**: Device connectivity
- **Kinesis**: Real-time data streaming

## Caveats

- By default, generates synthetic production data.
- Full AWS deployment requires AWS credentials and infrastructure setup.
- Production deployment requires additional security and monitoring.
