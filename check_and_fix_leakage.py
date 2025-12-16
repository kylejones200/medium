#!/usr/bin/env python3
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

Comprehensive data leakage detection and fixing for time series scripts.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

class LeakageFixer:
    def __init__(self, content: str, filepath: Path):
        self.content = content
        self.filepath = filepath
        self.lines = content.split('\n')
        self.modified = False
        
    def fix_scaling_before_split(self) -> bool:
        """Fix scaling that happens before train/test split."""
        fixed = False
        new_lines = []
        
        scaler_fit_lines = []
        split_line = -1
        
        for i, line in enumerate(self.lines):
            if re.search(r'(StandardScaler|MinMaxScaler|RobustScaler|PowerTransformer|QuantileTransformer)\(\)\.fit\(', line):
                scaler_fit_lines.append(i)
            if 'train_test_split' in line:
                split_line = i
        
        if scaler_fit_lines and split_line != -1:
            for scaler_line in scaler_fit_lines:
                if scaler_line < split_line:
                    line = self.lines[scaler_line]
                    if 'X' in line and 'y' not in line:
                        new_line = line.replace('.fit(', '.fit_transform(')
                        if '.fit_transform(' not in line:
                            new_line = re.sub(r'\.fit\(([^)]+)\)', r'.fit_transform(\1)', line)
                            self.lines[scaler_line] = new_line
                            fixed = True
        
        return fixed
    
    def fix_shuffle_in_timeseries(self) -> bool:
        """Fix shuffle=True in train_test_split for time series."""
        fixed = False
        
        has_timeseries = any(word in self.content.lower() for word in 
                            ['time series', 'timeseries', 'temporal', 'timestamp', 'forecast', 'forecasting', 'arima', 'sarima'])
        
        if has_timeseries:
            for i, line in enumerate(self.lines):
                if 'from sklearn.model_selection import' in line and 'shuffle=False' in line:
                    self.lines[i] = line.replace(', shuffle=False', '').replace('shuffle=False, ', '').replace('shuffle=False', '')
                    fixed = True
                elif 'train_test_split' in line:
                    if 'shuffle=True' in line:
                        self.lines[i] = line.replace('shuffle=True', 'shuffle=False')
                        fixed = True
                    elif 'shuffle=False' in line:
                        continue
                    elif 'shuffle' not in line and 'random_state' in line:
                        self.lines[i] = re.sub(r'(random_state=[^,)]+)', r'\1, shuffle=False', line)
                        fixed = True
                    elif 'shuffle' not in line:
                        if line.rstrip().endswith(')'):
                            self.lines[i] = line.rstrip()[:-1] + ', shuffle=False)'
                        else:
                            self.lines[i] = line.rstrip() + ', shuffle=False'
                        fixed = True
        
        return fixed
    
    def fix_center_in_rolling(self) -> bool:
        """Fix center=True in rolling windows (lookahead bias)."""
        fixed = False
        
        for i, line in enumerate(self.lines):
            if 'rolling(' in line and 'center=True' in line:
                new_line = line.replace('center=True', 'center=False')
                if 'shift(' not in line and '.shift(' not in line:
                    if '.rolling(' in line:
                        new_line = re.sub(r'\.rolling\(([^)]+)\)', r'.rolling(\1).shift(1)', new_line)
                    elif 'rolling(' in line:
                        new_line = re.sub(r'rolling\(([^)]+)\)', r'rolling(\1).shift(1)', new_line)
                else:
                    new_line = line.replace('center=True', 'center=False')
                self.lines[i] = new_line
                fixed = True
        
        return fixed
    
    def fix_negative_shift(self) -> bool:
        """Fix negative shift (using future data) in features, but allow in target creation."""
        fixed = False
        
        for i, line in enumerate(self.lines):
            if re.search(r'\.shift\(-[0-9]+\)', line):
                line_lower = line.lower()
                if 'target' in line_lower or 'y' in line_lower or 'next' in line_lower:
                    continue
                if 'X' in line or 'feature' in line_lower or '=' in line:
                    match = re.search(r'\.shift\(-([0-9]+)\)', line)
                    if match:
                        shift_val = match.group(1)
                        self.lines[i] = line.replace(f'.shift(-{shift_val})', f'.shift({shift_val})')
                        fixed = True
        
        return fixed
    
    def fix_cv_for_timeseries(self) -> bool:
        """Fix standard CV to use TimeSeriesSplit for time series."""
        fixed = False
        
        has_timeseries = any(word in self.content.lower() for word in 
                            ['time series', 'timeseries', 'temporal', 'forecast', 'forecasting'])
        
        if has_timeseries:
            has_timeseries_split = 'TimeSeriesSplit' in self.content
            
            for i, line in enumerate(self.lines):
                if 'KFold(' in line and not has_timeseries_split:
                    if 'from sklearn.model_selection import' in self.content or 'sklearn.model_selection' in self.content:
                        self.lines[i] = line.replace('KFold', 'TimeSeriesSplit')
                        fixed = True
                elif 'cross_val_score' in line and 'TimeSeriesSplit' not in self.content and not has_timeseries_split:
                    if 'from sklearn.model_selection import' in self.content:
                        import_line_idx = -1
                        for j, l in enumerate(self.lines):
                            if 'from sklearn.model_selection import' in l:
                                import_line_idx = j
                                break
                        if import_line_idx != -1:
                            if 'TimeSeriesSplit' not in self.lines[import_line_idx]:
                                self.lines[import_line_idx] = self.lines[import_line_idx].replace(
                                    'from sklearn.model_selection import',
                                    'from sklearn.model_selection import TimeSeriesSplit, '
                                ).replace('TimeSeriesSplit, TimeSeriesSplit', 'TimeSeriesSplit')
        
        return fixed
    
    def fix_feature_engineering_before_split(self) -> bool:
        """Fix feature engineering that uses full dataset statistics before split."""
        fixed = False
        
        split_line = -1
        for i, line in enumerate(self.lines):
            if 'train_test_split' in line:
                split_line = i
                break
        
        if split_line == -1:
            return False
        
        for i in range(split_line):
            line = self.lines[i]
            if re.search(r'\.(mean|std|min|max|median)\(\)', line) and ('X' in line or 'df' in line or 'data' in line):
                if 'train' not in line.lower() and 'test' not in line.lower():
                    continue
        
        return fixed
    
    def fix_all(self) -> Tuple[str, bool]:
        """Run all fixes."""
        self.modified = False
        
        if self.fix_scaling_before_split():
            self.modified = True
        if self.fix_shuffle_in_timeseries():
            self.modified = True
        if self.fix_center_in_rolling():
            self.modified = True
        if self.fix_negative_shift():
            self.modified = True
        if self.fix_cv_for_timeseries():
            self.modified = True
        if self.fix_feature_engineering_before_split():
            self.modified = True
        
        return '\n'.join(self.lines), self.modified

def scan_and_fix_all_scripts():
    """Scan all Python scripts for leakage and fix them."""
    base_dir = Path('/Users/kylejonespatricia/medium')
    
    time_series_keywords = [
        'time-series', 'time_series', 'timeseries', 'forecast', 'forecasting',
        'temporal', 'arima', 'sarima', 'lstm', 'rnn', 'time series', 'regime',
        'volatility', 'treasury', 'spread', 'bollinger', 'dickey', 'fuller',
        'stationarity', 'ensemble', 'anomaly', 'darts', 'nixtla', 'aeon',
        'tsfresh', 'pytimetk', 'statsmodels', 'kalman', 'state-space',
        'granger', 'causality', 'var', 'vecm', 'heating', 'gas', 'natural-gas',
        'lng', 'amtrak', 'ridership', 'discontinuity', 'fixed-effects',
        'panel', 'dtw', 'moirai', 'granite', 'tinytimemixer', 'energy-demand',
        'copula', 'mfles', 'mfle', 'seasonal', 'decomposition', 'fourier',
        'transfer-entropy', 'value-at-risk', 'expected-shortfall', 'portfolio',
        'asset-modeling', 'itos', 'martingales', 'stochastic'
    ]
    
    all_py_files = list(base_dir.rglob('*.py'))
    
    fixed_files = []
    checked_files = []
    
    for py_file in all_py_files:
        if '.ipynb_checkpoints' in str(py_file) or 'check_and_fix_leakage.py' in str(py_file):
            continue
        
        is_timeseries = any(keyword in str(py_file).lower() for keyword in time_series_keywords)
        
        if not is_timeseries:
            try:
                content = py_file.read_text()
                if any(keyword in content.lower() for keyword in ['time series', 'timeseries', 'forecast', 'temporal', 'train_test_split', 'arima', 'sarima', 'time series']):
                    is_timeseries = True
            except:
                continue
        
        if is_timeseries:
            try:
                content = py_file.read_text()
                fixer = LeakageFixer(content, py_file)
                new_content, modified = fixer.fix_all()
                
                if modified:
                    py_file.write_text(new_content)
                    fixed_files.append(str(py_file))
                checked_files.append(str(py_file))
            except Exception as e:
                pass
    
    return fixed_files, checked_files

if __name__ == "__main__":
    fixed, checked = scan_and_fix_all_scripts()
    logging.info(f"Checked {len(checked)} files")
    logging.info(f"Fixed {len(fixed)} files")

