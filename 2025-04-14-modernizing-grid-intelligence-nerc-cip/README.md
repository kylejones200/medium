# Modernizing Grid Intelligence with NERC CIP

This project demonstrates modernizing grid intelligence systems with NERC CIP (Critical Infrastructure Protection) compliance.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Grid intelligence functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- NERC CIP compliance options
- Output settings

## NERC CIP Compliance

NERC CIP standards:
- **CIP-002**: BES Cyber System Categorization
- **CIP-003**: Security Management Controls
- **CIP-004**: Personnel & Training
- **CIP-005**: Electronic Security Perimeters
- **CIP-006**: Physical Security
- **CIP-007**: Systems Security Management
- **CIP-008**: Incident Reporting
- **CIP-009**: Recovery Plans
- **CIP-010**: Configuration Change Management
- **CIP-011**: Information Protection

## Caveats

- By default, generates synthetic grid data.
- Full NERC CIP compliance requires comprehensive security implementation.
- Real-world deployment requires additional security measures.
