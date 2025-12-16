"""Core functions for Terraform HCL analysis."""

import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze_terraform_structure(terraform_dir: Path) -> Dict:
    """Analyze Terraform project structure."""
    files = list(terraform_dir.glob('*.tf'))
    vars_files = list(terraform_dir.glob('*.tfvars'))
    
    return {
        'total_files': len(files),
        'variable_files': len(vars_files),
        'file_names': [f.name for f in files]
    }

def validate_terraform_structure(structure: Dict) -> Dict:
    """Validate Terraform project structure."""
    required_files = ['main.tf', 'variables.tf', 'outputs.tf']
    found_files = structure['file_names']
    
    return {
        'has_main': 'main.tf' in found_files,
        'has_variables': 'variables.tf' in found_files,
        'has_outputs': 'outputs.tf' in found_files,
        'is_valid': all(req in found_files for req in required_files)
    }

def plot_terraform_structure(structure: Dict, validation: Dict, title: str, output_path: Path):
 """Plot Terraform structure analysis """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Total Files', 'Variable Files', 'Has Main', 'Has Variables', 'Has Outputs']
    values = [
        structure['total_files'],
        structure['variable_files'],
        1 if validation['has_main'] else 0,
        1 if validation['has_variables'] else 0,
        1 if validation['has_outputs'] else 0
    ]
    
    colors = ["#4A90A4" if v > 0 else "#D4A574" for v in values]
    ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='none')
    
    ax.set_ylabel("Count")
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

