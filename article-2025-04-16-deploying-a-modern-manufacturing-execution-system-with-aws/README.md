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
python main.py --data-path data/production_data.csv
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

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic production data.
- Full AWS deployment requires AWS credentials and infrastructure setup.
- Production deployment requires additional security and monitoring.
