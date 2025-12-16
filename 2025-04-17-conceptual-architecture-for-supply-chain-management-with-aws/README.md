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

## Caveats

- By default, generates synthetic supply chain data.
- Full AWS deployment requires AWS credentials and infrastructure setup.
- Real-world implementation requires integration with ERP systems.
