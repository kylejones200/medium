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

Run with default settings (generates synthetic IoT data):
```bash
python main.py
```

Run with your own data:
```bash
python main.py --data-path data/iot_data.csv
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

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic IoT sensor data.
- Full AWS pipeline requires AWS credentials and infrastructure setup.
- Real-time processing requires proper scaling configuration.
