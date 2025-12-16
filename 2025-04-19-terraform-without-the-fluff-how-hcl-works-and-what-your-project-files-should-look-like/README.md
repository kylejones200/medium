# Terraform Without the Fluff: How HCL Works and What Your Project Files Should Look Like

This project demonstrates Terraform HCL analysis and project structure validation.

## Article

Medium article: [Terraform Without the Fluff: How HCL Works and What Your Project Files Should Look Like](https://medium.com/@kylejones_47003/terraform-without-the-fluff-how-hcl-works-and-what-your-project-files-should-look-like-7e400c3813d2)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Terraform analysis functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Terraform project directory
- Structure validation options
- Output settings

## Terraform HCL

HCL (HashiCorp Configuration Language):
- **Declarative syntax**: Describes desired state
- **Blocks**: Resources, variables, outputs
- **Project structure**: Organized file structure
- **Best practices**: Standard patterns

## Caveats

- Analyzes Terraform file structure, not HCL syntax.
- Validation checks for standard file names.
- Full validation requires terraform validate command.
