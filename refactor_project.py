#!/usr/bin/env python3
"""
Tool to refactor a project from notebook-based to standardized structure.

Usage:
    python refactor_project.py <project_dir>
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def extract_code_from_notebook(notebook_path: Path) -> List[Tuple[str, str]]:
    """Extract Python code cells from a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if source.strip():
                code_cells.append((source, cell.get('execution_count')))
    
    return code_cells


def clean_code(code: str) -> str:
    """Remove excessive comments and clean up code."""
    lines = code.split('\n')
    cleaned = []
    skip_next = False
    
    for line in lines:
        stripped = line.strip()
        
        if skip_next:
            skip_next = False
            continue
            
        if stripped.startswith('"""') and stripped.endswith('"""') and len(stripped) > 3:
            continue
            
        if stripped.startswith("'''") and stripped.endswith("'''") and len(stripped) > 3:
            continue
            
        if re.match(r'^#+\s*$', stripped):
            continue
            
        if stripped.startswith('# ') and len(stripped) < 50:
            if any(word in stripped.lower() for word in ['import', 'load', 'fit', 'plot', 'save']):
                continue
        
        cleaned.append(line)
    
    return '\n'.join(cleaned)


def identify_functions(code: str) -> List[Dict]:
    """Identify function definitions in code."""
    functions = []
    pattern = r'def\s+(\w+)\s*\([^)]*\):'
    
    for match in re.finditer(pattern, code):
        func_name = match.group(1)
        start = match.start()
        end = code.find('\n\n', start)
        if end == -1:
            end = len(code)
        func_code = code[start:end]
        functions.append({
            'name': func_name,
            'code': func_code
        })
    
    return functions


def categorize_code_blocks(code_cells: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Categorize code into data loading, processing, modeling, plotting."""
    categories = {
        'data': [],
        'processing': [],
        'modeling': [],
        'plotting': [],
        'other': []
    }
    
    for code, _ in code_cells:
        code_lower = code.lower()
        
        if any(keyword in code_lower for keyword in ['read_csv', 'read_', 'load', 'fetch', 'download', 'pd.read']):
            categories['data'].append(code)
        elif any(keyword in code_lower for keyword in ['fit', 'train', 'model', 'predict', 'forecast', 'arima', 'sarima']):
            categories['modeling'].append(code)
        elif any(keyword in code_lower for keyword in ['plot', 'plt.', 'fig', 'savefig', 'show()']):
            categories['plotting'].append(code)
        elif any(keyword in code_lower for keyword in ['transform', 'scale', 'normalize', 'decompose', 'stationary', 'adf']):
            categories['processing'].append(code)
        else:
            categories['other'].append(code)
    
    return categories


def create_project_structure(project_dir: Path):
    """Create standardized project structure."""
    dirs = ['src', 'tests', 'data', 'images']
    for d in dirs:
        (project_dir / d).mkdir(exist_ok=True)
    
    files = {
        'main.py': '',
        'config.yaml': '',
        'requirements.txt': project_dir / 'requirements.txt' if (project_dir / 'requirements.txt').exists() else None
    }
    
    return dirs, files


def generate_main_py(categories: Dict[str, List[str]], project_name: str) -> str:
    """Generate main.py file."""
    template = f'''#!/usr/bin/env python3
"""
Main entry point for {project_name}.

Run with: python main.py
"""

import argparse
import yaml
from pathlib import Path
from src.core import *


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='{project_name}')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--output-dir', type=Path, default=Path('images'), help='Output directory for plots')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # TODO: Wire up functions from src/core.py
    print("Running {project_name}...")
    print(f"Config: {{config}}")
    print(f"Output directory: {{output_dir}}")


if __name__ == "__main__":
    main()
'''
    return template


def generate_config_yaml() -> str:
    """Generate default config.yaml."""
    return '''# Configuration file
# Adjust these values as needed

data:
  source: null  # Path to data file or URL
  n_samples: 200
  start_date: "2023-01-01"
  frequency: "D"

model:
  arima_order: [2, 1, 2]
  hold_out_days: 30

output:
  figures_dir: "images"
  figure_format: "png"
  figure_dpi: 100
'''


def refactor_project(project_dir: Path):
    """Refactor a project to standardized structure."""
    project_dir = Path(project_dir)
    
    if not project_dir.exists():
        print(f"Error: {project_dir} does not exist")
        return
    
    print(f"Refactoring project: {project_dir.name}")
    
    notebooks = list(project_dir.glob('*.ipynb'))
    if not notebooks:
        print("No notebooks found in project directory")
        return
    
    print(f"Found {len(notebooks)} notebook(s)")
    
    create_project_structure(project_dir)
    
    all_code = []
    for nb_path in notebooks:
        print(f"  Processing {nb_path.name}...")
        code_cells = extract_code_from_notebook(nb_path)
        all_code.extend(code_cells)
    
    categories = categorize_code_blocks(all_code)
    
    print("\nCode categorization:")
    for cat, blocks in categories.items():
        print(f"  {cat}: {len(blocks)} blocks")
    
    main_py = generate_main_py(categories, project_dir.name)
    config_yaml = generate_config_yaml()
    
    (project_dir / 'main.py').write_text(main_py)
    (project_dir / 'config.yaml').write_text(config_yaml)
    
    print(f"\nCreated main.py and config.yaml in {project_dir}")
    print("Next steps:")
    print("  1. Review and extract functions to src/core.py")
    print("  2. Update main.py to wire up the functions")
    print("  3. Add tests to tests/")
    print("  4. Update README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Refactor project to standardized structure')
    parser.add_argument('project_dir', type=Path, help='Project directory to refactor')
    args = parser.parse_args()
    
    refactor_project(args.project_dir)

