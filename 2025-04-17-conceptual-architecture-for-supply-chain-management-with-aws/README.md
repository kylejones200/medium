# Conceptual Architecture for Supply Chain Management with AWS

This project demonstrates conceptual architecture for supply chain management using AWS.

## Article

Medium article: [Conceptual Architecture for Supply Chain Management with AWS](https://medium.com/@kylejones_47003/conceptual-architecture-for-supply-chain-management-with-aws-a302638fac0f)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Supply chain functions
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
python main.py --data-path data/supply_chain_data.csv
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Number of supply chain nodes
- AWS services configuration
- Output settings

## AWS Architecture

AWS services for supply chain:
- **EventBridge**: Event-driven architecture
- **Step Functions**: Workflow orchestration
- **DynamoDB**: NoSQL database
- **Lambda**: Serverless compute
- **S3**: Data storage

## Plotting Style

All plots use a minimalist Tufte-style design:
- No gridlines
- No top or right spines
- Descriptive titles
- Muted, professional color palette

## Caveats

- By default, generates synthetic supply chain data.
- Full AWS deployment requires AWS credentials and infrastructure setup.
- Real-world implementation requires integration with ERP systems.
