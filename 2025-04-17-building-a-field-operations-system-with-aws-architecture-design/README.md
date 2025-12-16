# Building a Field Operations System with AWS Architecture Design

This project demonstrates building a field operations system using AWS architecture.

## Article

Medium article: [Building a Field Operations System with AWS Architecture Design](https://medium.com/@kylejones_47003/building-a-field-operations-system-with-aws-architecture-design-121257004c38)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Field operations functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Number of assets
- AWS services configuration
- Output settings

## AWS Architecture

AWS services for field operations:
- **IoT Core**: Device connectivity
- **ECS**: Container orchestration
- **RDS**: Database management
- **S3**: Data storage
- **CloudWatch**: Monitoring and logging

## Caveats

- By default, generates synthetic field operations data.
- Full AWS deployment requires AWS credentials and infrastructure setup.
- Real-world implementation requires proper security and scaling.
