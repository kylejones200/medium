#!/usr/bin/env python3
"""
Terraform Without the Fluff: How HCL Works and What Your Project Files Should Look Like

Main entry point for running Terraform HCL analysis.

Usage:
    python main.py
    python main.py --terraform-dir terraform/
"""

import argparse
import yaml
from pathlib import Path
from src.core import analyze_terraform_structure, validate_terraform_structure, plot_terraform_structure


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Terraform HCL Analysis')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--terraform-dir', type=Path, default=None, help='Terraform project directory')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    terraform_dir = Path(args.terraform_dir) if args.terraform_dir else Path(config['terraform']['project_dir'])
    
    print(f"Analyzing Terraform project structure in {terraform_dir}...")
    structure = analyze_terraform_structure(terraform_dir)
    
    print(f"\nTerraform Structure:")
    print(f"Total .tf files: {structure['total_files']}")
    print(f"Variable files: {structure['variable_files']}")
    print(f"Files found: {', '.join(structure['file_names'])}")
    
    if config['terraform']['validate_structure']:
        validation = validate_terraform_structure(structure)
        print(f"\nStructure Validation:")
        print(f"Has main.tf: {validation['has_main']}")
        print(f"Has variables.tf: {validation['has_variables']}")
        print(f"Has outputs.tf: {validation['has_outputs']}")
        print(f"Valid structure: {validation['is_valid']}")
        
        if not validation['is_valid']:
            print("\n⚠️  Warning: Missing required Terraform files")
        
        plot_terraform_structure(structure, validation, "Terraform Project Structure Analysis",
                               output_dir / 'terraform_structure.png')
    
    print(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()

