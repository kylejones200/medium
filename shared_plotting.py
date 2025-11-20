"""
Shared Tufte-style plotting utilities for all projects.

Copy this to your project's src/plotting.py or import from a shared location.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional


def setup_tufte_style():
    """Configure matplotlib to use Tufte-style minimalism.
    
    Features:
    - No gridlines
    - No top and right spines
    - Muted colors
    - Descriptive titles
    - Clean, minimal design
    """
    try:
        plt.style.use('seaborn-v0_8-white')
    except OSError:
        try:
            plt.style.use('seaborn-white')
        except OSError:
            plt.style.use('default')
    
    mpl.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '0.3',
        'axes.linewidth': 0.8,
        'xtick.color': '0.3',
        'ytick.color': '0.3',
        'text.color': '0.2',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.frameon': False,
        'legend.fontsize': 9,
        'figure.dpi': 100,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
    })


def apply_tufte_style(ax=None, title: Optional[str] = None):
    """Apply Tufte style to current or specified axes.
    
    Args:
        ax: Matplotlib axes object (uses current if None)
        title: Optional descriptive title for the plot
    """
    if ax is None:
        ax = plt.gca()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    
    if title:
        ax.set_title(title, pad=10)
    
    return ax


def save_tufte_figure(output_path, dpi: int = 100, bbox_inches: str = 'tight'):
    """Save figure with Tufte style settings.
    
    Args:
        output_path: Path to save the figure
        dpi: Resolution for saved figure
        bbox_inches: Bounding box setting
    """
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
    plt.close()

