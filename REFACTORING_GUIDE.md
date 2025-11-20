# Repository Refactoring Guide

This guide documents the standardized structure and refactoring process for converting notebook-based projects to clean, reproducible Python code.

## Standard Structure

Each project should follow this structure:

```
project-name/
├── README.md           # Explains the article, how to run, caveats
├── main.py             # Clear entry point with CLI
├── config.yaml         # Configuration (no hard-coded values)
├── requirements.txt    # Python dependencies
├── src/                # Pure functions, reusable code
│   ├── core.py         # Main logic as testable functions
│   └── plotting.py     # Tufte-style plotting utilities
├── tests/              # Pytest tests
│   └── test_core.py    # Unit tests for core functions
├── data/               # Local data files (if needed)
└── images/             # Generated plots and figures
```

## Principles

1. **One command to run**: `python main.py` should work from a clean environment
2. **No hard-coded paths**: Use config.yaml and CLI arguments
3. **Clear separation**: Config, data fetch, logic, and plots are separate
4. **Testable functions**: Pure functions in `src/` that take args and return results
5. **Zero hidden state**: No global variables, all state passed explicitly

## Refactoring Steps

### 1. Create Directory Structure

```bash
mkdir -p src tests data images
```

### 2. Extract Logic to `src/core.py` and `src/plotting.py`

- Convert notebook cells to pure functions
- Functions should take parameters and return results
- Remove excessive comments (code should be self-describing)
- Remove hard-coded values (move to config)
- Create `src/plotting.py` for Tufte-style plotting utilities

Example:
```python
# Bad (from notebook):
df = pd.read_csv("/Users/me/data.csv")  # Hard-coded path
model.fit(train)  # Hard-coded data

# Good (in src/core.py):
def load_data(data_path: Path) -> pd.DataFrame:
    """Load time series data from file."""
    return pd.read_csv(data_path, parse_dates=['date'], index_col='date')

def fit_model(train_data: pd.Series, order: Tuple[int, int, int]) -> Model:
    """Fit ARIMA model to training data."""
    model = ARIMA(train_data, order=order)
    return model.fit()
```

### 3. Create `src/plotting.py` for Tufte Style

Create a plotting utility module with Tufte-style minimalism:

```python
# src/plotting.py
def setup_tufte_style():
    """Configure matplotlib for Tufte-style: no gridlines, no top/right spines."""
    mpl.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        # ... other settings
    })

def apply_tufte_style(ax, title=None):
    """Apply Tufte style to axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    if title:
        ax.set_title(title, pad=10)
```

Then use in plotting functions:
```python
from .plotting import setup_tufte_style, apply_tufte_style, save_tufte_figure

def plot_forecast(data, output_path):
    setup_tufte_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data.values, linewidth=1.2)
    apply_tufte_style(ax, title="Descriptive Title with Key Metrics")
    save_tufte_figure(output_path)
```

### 4. Create `config.yaml`

Move all configuration values to YAML:

```yaml
data:
  source: null  # Path to data or null for synthetic
  n_samples: 200

model:
  arima_order: [2, 1, 2]
  hold_out_days: 30

output:
  figures_dir: "images"
```

### 5. Create `main.py`

- Load config from YAML
- Parse CLI arguments
- Wire up functions from `src/core.py`
- Handle data loading, processing, and output

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--data-path', type=Path, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    df = load_data(args.data_path or config['data']['source'])
    # ... wire up functions
```

### 6. Create Tests

Write pytest tests for core functions:

```python
def test_load_data():
    df = load_data(Path("test_data.csv"))
    assert len(df) > 0
    assert 'value' in df.columns
```

### 7. Update README.md

Include:
- Article link
- Project structure explanation
- Setup instructions
- Usage examples
- Caveats and known issues

### 8. Create Thin Notebook (Optional)

If you still want a notebook for visualization, make it a thin wrapper:

```python
# In notebook
from src.core import *
import yaml

config = yaml.safe_load(open('config.yaml'))
df = load_data()
# ... call functions from src/core.py
```

## Code Quality Guidelines

1. **Remove excessive comments**: Code should be self-describing
2. **Use type hints**: `def process_data(df: pd.DataFrame) -> pd.Series:`
3. **Pure functions**: No side effects, return values instead of printing
4. **Idiomatic Python**: Use pathlib, f-strings, context managers
5. **Error handling**: Use exceptions, not silent failures

## Example: Complete Refactoring

See `article-2025-01-12-time-series-analysis-with-statsmodels-in-python/` for a complete example.

## Automation

Use `refactor_project.py` to get started:

```bash
python refactor_project.py article-2025-01-12-time-series-analysis-with-statsmodels-in-python
```

This creates the structure and extracts code, but you'll need to:
1. Review and clean up `src/core.py`
2. Wire up functions in `main.py`
3. Add tests
4. Update README

## Checklist

- [ ] Directory structure created
- [ ] Logic extracted to `src/core.py` as pure functions
- [ ] `config.yaml` created with all configuration
- [ ] `main.py` wires everything together
- [ ] Tests written in `tests/`
- [ ] README.md updated with usage instructions
- [ ] `requirements.txt` includes all dependencies
- [ ] Code runs with `python main.py` from clean environment
- [ ] No hard-coded paths
- [ ] No global variables
- [ ] Excessive comments removed

