#!/usr/bin/env python3
"""
Security check script to detect hardcoded API keys and secrets.

This script should be run before committing to prevent accidental exposure
of API keys, tokens, and other sensitive information.
"""

import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Common API key patterns
API_KEY_PATTERNS = [
    (r'api[_-]?key\s*[:=]\s*["\']([a-zA-Z0-9]{20,})["\']', 'API Key'),
    (r'apikey\s*[:=]\s*["\']([a-zA-Z0-9]{20,})["\']', 'API Key'),
    (r'["\']([a-zA-Z0-9]{32,})["\']', 'Potential API Key (long alphanumeric)'),
]

# Service-specific patterns
SERVICE_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{32,}', 'OpenAI API Key'),
    (r'pk_[a-zA-Z0-9]{32,}', 'Stripe Publishable Key'),
    (r'sk_live_[a-zA-Z0-9]{32,}', 'Stripe Secret Key'),
    (r'AIza[0-9A-Za-z_-]{35}', 'Google API Key'),
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key ID'),
    (r'ghp_[a-zA-Z0-9]{36}', 'GitHub Personal Access Token'),
    (r'gho_[a-zA-Z0-9]{36}', 'GitHub OAuth Token'),
    (r'xoxb-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}', 'Slack Bot Token'),
    (r'xoxa-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}', 'Slack App Token'),
    (r'xoxp-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}', 'Slack User Token'),
]

# FRED API key pattern (32 hex characters)
FRED_PATTERN = (r'["\']([0-9a-f]{32})["\']', 'FRED API Key')

# Files to exclude
EXCLUDE_PATTERNS = [
    r'\.git/',
    r'__pycache__/',
    r'\.pyc$',
    r'\.ipynb_checkpoints/',
    r'venv/',
    r'env/',
    r'\.venv/',
    r'node_modules/',
    r'\.gitignore',
    r'check_secrets\.py',
    r'\.env',
]

def should_exclude_file(file_path: Path) -> bool:
    """Check if file should be excluded from scanning."""
    path_str = str(file_path)
    return any(re.search(pattern, path_str) for pattern in EXCLUDE_PATTERNS)

def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan a file for potential secrets."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return [(0, f"Error reading file: {e}", "")]
    
    # Check for API key patterns
    for line_num, line in enumerate(lines, 1):
        # Check generic API key patterns
        for pattern, description in API_KEY_PATTERNS:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                # Skip if it's clearly a placeholder
                key_value = match.group(1) if match.groups() else match.group(0)
                if any(placeholder in key_value.lower() for placeholder in ['your_key', 'placeholder', 'example', 'xxx', 'test']):
                    continue
                issues.append((line_num, description, line.strip()))
        
        # Check service-specific patterns
        for pattern, description in SERVICE_PATTERNS:
            if re.search(pattern, line):
                issues.append((line_num, description, line.strip()))
        
        # Check FRED API key pattern (but exclude if it's in a comment or clearly a placeholder)
        if 'fred' in line.lower() or 'api' in line.lower():
            fred_match = re.search(FRED_PATTERN[0], line, re.IGNORECASE)
            if fred_match:
                key_value = fred_match.group(1)
                if not any(placeholder in line.lower() for placeholder in ['your_key', 'placeholder', 'example', 'xxx']):
                    issues.append((line_num, FRED_PATTERN[1], line.strip()))
    
    return issues

def scan_directory(directory: Path) -> List[Tuple[Path, List[Tuple[int, str, str]]]]:
    """Scan a directory for files with potential secrets."""
    results = []
    
    # Files to scan
    extensions = ['.py', '.yaml', '.yml', '.json', '.env', '.config', '.conf']
    
    for file_path in directory.rglob('*'):
        if file_path.is_file() and not should_exclude_file(file_path):
            if any(file_path.suffix == ext for ext in extensions):
                issues = scan_file(file_path)
                if issues:
                    results.append((file_path, issues))
    
    return results

def main():
    """Main function to run security checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check for hardcoded API keys and secrets')
    parser.add_argument('--path', type=Path, default=Path('.'), help='Path to scan (default: current directory)')
    parser.add_argument('--exit-on-find', action='store_true', help='Exit with error code if secrets found')
    args = parser.parse_args()
    
    scan_path = Path(args.path)
    if not scan_path.exists():
        logging.error(f"Error: Path {scan_path} does not exist")
        sys.exit(1)
    
    logging.info(f"Scanning {scan_path} for potential secrets...")
    
    if scan_path.is_file():
        issues = scan_file(scan_path)
        if issues:
            logging.warning(f"Found {len(issues)} potential issue(s) in {scan_path}:")
            for line_num, description, line in issues:
                logging.warning(f"  Line {line_num}: {description}")
                logging.warning(f"    {line[:100]}")
        else:
            logging.info(f"✓ No issues found in {scan_path}")
    else:
        results = scan_directory(scan_path)
        
        if results:
            logging.warning(f"Found potential secrets in {len(results)} file(s):")
            for file_path, issues in results:
                logging.warning(f"📄 {file_path}")
                for line_num, description, line in issues:
                    logging.warning(f"   Line {line_num}: {description}")
                    logging.warning(f"   {line[:100]}")
            
            if args.exit_on_find:
                logging.error("❌ Security check failed. Please remove hardcoded secrets before committing.")
                sys.exit(1)
        else:
            logging.info("✓ No hardcoded secrets found!")

if __name__ == "__main__":
    main()

